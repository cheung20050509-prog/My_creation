#!/usr/bin/env python3
"""Model complexity metrics for fixed_experiment frozen trials (subprocess per case).

Defaults to conda env **ITHP5090**: per-case ``--measure-worker`` subprocesses use
that interpreter (see ``_worker_python()``). Set env **MODEL_COMPLEXITY_PYTHON** to
override. Launch the driver with **ITHP5090** as well (``run_measure.sh`` or the full
``python`` path below) so imports match workers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_MY_CREATION = _HERE.parent
_FIXED_EXP = _MY_CREATION / "fixed_experiment"
_TRAIN_PY = _FIXED_EXP / "train.py"

# Same default interpreter as ``run_optuna_4090d_restart.sh`` on this project layout.
_DEFAULT_ITHP5090_PY = Path("/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python")


def _worker_python() -> str:
    """Python for ``--measure-worker`` subprocesses; defaults to ITHP5090."""
    override = os.environ.get("MODEL_COMPLEXITY_PYTHON", "").strip()
    if override and Path(override).is_file():
        return override
    if _DEFAULT_ITHP5090_PY.is_file():
        return str(_DEFAULT_ITHP5090_PY)
    return sys.executable


_CASE_TO_HP = {
    "mosi": _FIXED_EXP / "mosi_trial234_hparams.py",
    "mosei": _FIXED_EXP / "mosei_phase1_trial70_hparams.py",
    "simsv2": _FIXED_EXP / "simsv2_phase4_trial52_hparams.py",
}


def _load_build_train_argv(hp_py: Path):
    spec = importlib.util.spec_from_file_location(
        f"_hparams_{hp_py.stem}", hp_py,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.build_train_argv


def _load_fixed_train_module():
    """Parse CLI from sys.argv and load fixed_experiment/train.py (must run after sys.argv is set)."""
    spec = importlib.util.spec_from_file_location(
        "fixed_experiment_train_meas", _TRAIN_PY,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _FlopMeasuringWrapper(nn.Module):
    """Expose a single forward(args...) for fvcore/thop; includes IB path (stage=1)."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        input_ids: torch.Tensor,
        visual: torch.Tensor,
        acoustic: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits, ib_loss, _, _ = self.core(
            input_ids, visual, acoustic, labels=labels, stage=1,
        )
        return logits.sum() + ib_loss


def _count_flops(wrapper: nn.Module, batch: tuple[torch.Tensor, ...]) -> tuple[float | None, str, str | None]:
    input_ids, visual, acoustic, labels = batch
    inputs = (input_ids, visual, acoustic, labels)
    err: str | None = None
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        was_training = wrapper.training
        wrapper.train()
        try:
            total = FlopCountAnalysis(wrapper, inputs).total()
        finally:
            wrapper.train(was_training)
        return float(total), "fvcore", None
    except Exception as e:  # noqa: BLE001
        err = f"fvcore:{e!r}"
    try:
        from thop import profile  # type: ignore

        wrapper.train()
        macs, _params = profile(wrapper, inputs=inputs, verbose=False)
        flops = 2.0 * float(macs)
        return flops, "thop", None
    except Exception as e2:  # noqa: BLE001
        err = (err or "") + f"|thop:{e2!r}"
    return None, "none", err


def _microbatch_from_loader(train_dl, device: torch.device):
    batch = next(iter(train_dl))
    batch = tuple(t.to(device) for t in batch)
    input_ids, visual, acoustic, label_ids = batch
    visual = visual.squeeze(1)
    acoustic = acoustic.squeeze(1)
    return input_ids, visual, acoustic, label_ids


def _one_step_loss(model, args, input_ids, visual, acoustic, label_ids, stage: int):
    logits, ib_loss, loss_dict, _ = model(
        input_ids, visual, acoustic, labels=label_ids, stage=stage,
    )
    pred_flat = logits.view(-1)
    label_flat = label_ids.view(-1)
    loss_dict["pred_mean"] = pred_flat.mean().item()
    loss_dict["pred_std"] = pred_flat.std(unbiased=False).item()
    l_task = F.l1_loss(pred_flat, label_flat) + args.mse_weight * F.mse_loss(
        pred_flat, label_flat,
    )
    loss = l_task + ib_loss
    if args.gradient_accumulation_step > 1:
        loss = loss / args.gradient_accumulation_step
    return loss


def _peak_memory_microbatch(ft, model, optimizer, train_dl) -> float | None:
    if not torch.cuda.is_available():
        return None
    device = ft.DEVICE
    model.train()
    optimizer.zero_grad()
    input_ids, visual, acoustic, label_ids = _microbatch_from_loader(train_dl, device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    loss = _one_step_loss(model, ft.args, input_ids, visual, acoustic, label_ids, stage=1)
    loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0 ** 3)


def _peak_memory_accum_window(ft, model, optimizer, train_dl) -> float | None:
    if not torch.cuda.is_available():
        return None
    device = ft.DEVICE
    args = ft.args
    model.train()
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    it = iter(train_dl)
    ga = max(1, int(args.gradient_accumulation_step))
    for step in range(ga):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)
        batch = tuple(t.to(device) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)
        loss = _one_step_loss(model, ft.args, input_ids, visual, acoustic, label_ids, stage=1)
        loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0 ** 3)


def _dummy_batch_for_flops(ft, model):
    args = ft.args
    device = ft.DEVICE
    B = int(args.train_batch_size)
    L = int(args.max_seq_length)
    mod = ft
    vdim = mod.VISUAL_DIM
    adim = mod.ACOUSTIC_DIM
    input_ids = torch.zeros((B, L), dtype=torch.long, device=device)
    visual = torch.randn(B, L, vdim, device=device, dtype=torch.float32)
    acoustic = torch.randn(B, L, adim, device=device, dtype=torch.float32)
    labels = torch.randn(B, device=device, dtype=torch.float32)
    return input_ids, visual, acoustic, labels


def measure_worker(
    case: str,
    memory_full_accum: bool,
    *,
    skip_epoch_time: bool = False,
) -> dict[str, Any]:
    os.environ.setdefault("TQDM_DISABLE", "1")
    if str(_MY_CREATION) not in sys.path:
        sys.path.insert(0, str(_MY_CREATION))

    hp_path = _CASE_TO_HP[case]
    build_train_argv = _load_build_train_argv(hp_path)
    with tempfile.TemporaryDirectory() as tmp:
        argv = build_train_argv(checkpoint_dir=tmp)
    sys.argv = ["fixed_experiment/train.py", *argv]

    ft = _load_fixed_train_module()
    args = ft.args
    train_dl, _dev_dl, _test_dl, n_opt = ft.setup_data()
    model, optimizer, scheduler = ft.build_model(n_opt)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # CUDA warmup (not counted toward peak memory).
    if torch.cuda.is_available():
        optimizer.zero_grad()
        b0 = _microbatch_from_loader(train_dl, ft.DEVICE)
        loss0 = _one_step_loss(model, args, *b0, stage=1)
        loss0.backward()
        optimizer.zero_grad()
        torch.cuda.synchronize()

    # Peak memory: one microbatch or a full accum window (after warmup).
    if memory_full_accum:
        peak_gib = _peak_memory_accum_window(ft, model, optimizer, train_dl)
    else:
        peak_gib = _peak_memory_microbatch(ft, model, optimizer, train_dl)

    n_batches = len(train_dl)
    epoch_sec: float | None
    if skip_epoch_time:
        epoch_sec = None
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ft.train_epoch(model, train_dl, optimizer, scheduler, stage=1, ema=None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_sec = time.perf_counter() - t0

    wrap = _FlopMeasuringWrapper(model)
    flop_batch = _dummy_batch_for_flops(ft, model)
    flops, flop_backend, flop_err = _count_flops(wrap, flop_batch)

    return {
        "case": case,
        "dataset": args.dataset,
        "train_batch_size": int(args.train_batch_size),
        "gradient_accumulation_step": int(args.gradient_accumulation_step),
        "max_seq_length": int(args.max_seq_length),
        "params_total": int(total_p),
        "params_trainable": int(train_p),
        "flops_forward": flops,
        "flops_backend": flop_backend,
        "flops_error": flop_err,
        "peak_mem_gib": peak_gib,
        "memory_mode": "accum_window" if memory_full_accum else "single_microbatch",
        "epoch_sec": epoch_sec,
        "train_batches_per_epoch": int(n_batches),
    }


def _run_worker_subprocess(
    case: str,
    memory_full_accum: bool,
    skip_epoch_time: bool,
) -> dict[str, Any]:
    cmd = [
        _worker_python(),
        "-u",
        str(Path(__file__).resolve()),
        "--measure-worker",
        case,
    ]
    if memory_full_accum:
        cmd.append("--memory-full-accum")
    if skip_epoch_time:
        cmd.append("--skip-epoch-time")
    proc = subprocess.run(
        cmd,
        cwd=str(_MY_CREATION),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker {case} failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"worker {case}: empty stdout\nstderr:\n{proc.stderr}")
    return json.loads(lines[-1])


def _md_row(r: dict[str, Any]) -> str:
    def fmt_flops(x: Any) -> str:
        if x is None:
            return "—"
        xf = float(x)
        if xf >= 1e12:
            return f"{xf/1e12:.2f}T"
        if xf >= 1e9:
            return f"{xf/1e9:.2f}G"
        if xf >= 1e6:
            return f"{xf/1e6:.2f}M"
        return f"{xf:.2e}"

    flops_s = fmt_flops(r.get("flops_forward"))
    if r.get("flops_backend") == "none" and r.get("flops_error"):
        flops_s = f"— ({r['flops_error'][:80]}…)" if len(str(r.get("flops_error"))) > 80 else f"— ({r.get('flops_error')})"

    mem = r.get("peak_mem_gib")
    mem_s = f"{mem:.3f}" if mem is not None else "—"

    es = r.get("epoch_sec")
    epoch_s = f"{float(es):.2f}" if es is not None else "—"

    return (
        f"| {r['case']} | {r['params_total']:,} | {r['params_trainable']:,} | {flops_s} | "
        f"{mem_s} | {epoch_s} | {r['train_batches_per_epoch']} | "
        f"{r['train_batch_size']}×{r['gradient_accumulation_step']} | {r['max_seq_length']} |"
    )


def _print_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| case | params | trainable | FLOPs (1 fwd, est.) | peak GPU GiB | epoch_sec | "
        "batches/epoch | B×accum | seq |"
    )
    sep = "|---|---:|---:|---|---:|---:|---:|---|---:|"
    lines = [header, sep] + [_md_row(r) for r in rows]
    text = "\n".join(lines) + "\n"
    print(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--measure-worker",
        metavar="CASE",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--memory-full-accum",
        action="store_true",
        help="Peak memory over a full gradient-accumulation window (no optimizer.step).",
    )
    ap.add_argument(
        "--skip-epoch-time",
        action="store_true",
        help="Skip full train_epoch timing in the worker (faster smoke test).",
    )
    ap.add_argument(
        "--cases",
        default="mosi,mosei,simsv2",
        help="Comma-separated subset of mosi,mosei,simsv2 (driver only).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write Markdown table to this path.",
    )
    args = ap.parse_args()

    if args.measure_worker:
        case = args.measure_worker.strip().lower()
        if case not in _CASE_TO_HP:
            print(f"unknown case {case!r}", file=sys.stderr)
            return 2
        result = measure_worker(
            case,
            memory_full_accum=args.memory_full_accum,
            skip_epoch_time=args.skip_epoch_time,
        )
        print(json.dumps(result), flush=True)
        return 0

    cases = [c.strip().lower() for c in args.cases.split(",") if c.strip()]
    for c in cases:
        if c not in _CASE_TO_HP:
            print(f"unknown case in --cases: {c!r}", file=sys.stderr)
            return 2

    rows: list[dict[str, Any]] = []
    for c in cases:
        rows.append(
            _run_worker_subprocess(
                c,
                memory_full_accum=args.memory_full_accum,
                skip_epoch_time=args.skip_epoch_time,
            ),
        )

    table = _print_table(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        intro = (
            "# Model complexity (fixed_experiment)\n\n"
            "Auto-generated by `measure_fixed_cases.py`. "
            "See [README.md](README.md) for definitions.\n\n"
        )
        args.output.write_text(intro + table, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
