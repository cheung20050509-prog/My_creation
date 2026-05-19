#!/usr/bin/env python3
"""Verify SIMSv2 phase4 trial-52 hyperparameters match ``optuna_search_v2.objective()`` formatting.

Gold train log header (``logs/optuna/4090D_restart/phase4/train_logs/simsv2_phase4_trial_52.log``):
  LR (backbone)  : 2.019142e-05
  LR (InfoGate)  : 0.0003523834
  EMA: decay=0.9994724074207356, start_epoch=3

These match the second ``Trial 52 finished`` block in ``phase4/run/simsv2.log`` (paper PRISM row)
and align with ``verify_simsv2_phase6_trial0_optuna_train_argv.py`` embedded params.
"""

from __future__ import annotations

import argparse
import sys

TRIAL_52_PARAMS = {
    "n_epochs": 57,
    "learning_rate": 2.0191415275580603e-05,
    "ig_learning_rate": 0.00035238338811010466,
    "beta_ib": 5.440630703880903,
    "num_infogate_layers": 2,
    "bottleneck_dim": 64,
    "mse_weight": 0.7208657261411349,
    "dropout_prob": 0.06418453238018118,
    "alpha_ib": 0.004565520177989885,
    "stage1_epochs": 5,
    "warmup_proportion": 0.07839760585460172,
    "weight_decay": 0.0005782782842106815,
    "ema_decay": 0.9994724074207356,
    "selector_target_temp": 0.40925294131460815,
    "selector_rib_weight": 0.03961433930428535,
    "align_mix_floor": 0.3,
    "gumbel_tau_start": 1.6775306159158943,
    "gumbel_tau_end": 0.5644303121448456,
    "num_heads": 8,
    "unified_dim": 384,
    "ema_start_epoch": 3,
    "seed": 128,
}

GOLD_BANNER_LR_BACKBONE = "  LR (backbone)  : 2.019142e-05"
GOLD_BANNER_LR_INFOGATE = "  LR (InfoGate)  : 0.0003523834"
GOLD_EMA_LINE = "EMA: decay=0.9994724074207356, start_epoch=3"


def format_train_float_argv(p: dict) -> dict[str, str]:
    """Mirror optuna_search_v2.objective() argv formatting for float-like knobs."""
    return {
        "learning_rate": f"{p['learning_rate']:.6e}",
        "ig_learning_rate": f"{p['ig_learning_rate']:.6e}",
        "beta_ib": f"{p['beta_ib']:.4f}",
        "mse_weight": f"{p['mse_weight']:.4f}",
        "dropout_prob": f"{p['dropout_prob']:.4f}",
        "alpha_ib": f"{p['alpha_ib']:.6f}",
        "warmup_proportion": f"{p['warmup_proportion']:.4f}",
        "weight_decay": f"{p['weight_decay']:.6f}",
        "ema_decay": str(p["ema_decay"]),
        "selector_target_temp": f"{p['selector_target_temp']:.4f}",
        "selector_rib_weight": f"{p['selector_rib_weight']:.4f}",
        "align_mix_floor": f"{p['align_mix_floor']:.4f}",
        "gumbel_tau_start": f"{p['gumbel_tau_start']:.4f}",
        "gumbel_tau_end": f"{p['gumbel_tau_end']:.4f}",
    }


def banner_lines_after_parse(fmt: dict[str, str], ema_start_epoch: int) -> tuple[str, str, str]:
    lr = float(fmt["learning_rate"])
    ig = float(fmt["ig_learning_rate"])
    ema_f = float(fmt["ema_decay"])
    return (
        f"  LR (backbone)  : {lr}",
        f"  LR (InfoGate)  : {ig}",
        f"EMA: decay={ema_f}, start_epoch={ema_start_epoch}",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Exit without printing (still non-zero on failure).",
    )
    ap.add_argument(
        "--storage",
        default="sqlite:///logs/optuna/4090D_restart/phase4/db/simsv2.db",
        help="Optional sqlite URI; if trial 52 exists, merge params from DB.",
    )
    ap.add_argument(
        "--study-name",
        default="infogate_simsv2_phase4_4090d",
        help="Study name for --storage.",
    )
    args = ap.parse_args()

    params = dict(TRIAL_52_PARAMS)
    if args.storage:
        try:
            import optuna  # type: ignore

            study = optuna.load_study(study_name=args.study_name, storage=args.storage)
            t = next(
                (x for x in study.get_trials(deepcopy=False) if x.number == 52), None
            )
            if t is None or not t.params:
                if not args.quiet:
                    print(
                        "Note: trial 52 not in DB; using embedded TRIAL_52_PARAMS.",
                        file=sys.stderr,
                    )
            else:
                dbp = dict(t.params)
                bc = dbp.pop("batch_config", None)
                if bc is not None and bc != 0:
                    print(
                        f"WARNING: expected batch_config 0, got {bc}",
                        file=sys.stderr,
                    )
                for k, v in dbp.items():
                    if k in params and isinstance(params[k], float) and isinstance(v, float):
                        if abs(params[k] - v) > 1e-12 * max(1.0, abs(params[k])):
                            print(
                                f"WARNING: embedded {k}={params[k]} vs DB {v}",
                                file=sys.stderr,
                            )
                    params[k] = v
                if not args.quiet:
                    print(
                        "Loaded trial 52 params from Optuna storage "
                        f"(batch_config={bc}); defaults unchanged for non-DB keys."
                    )
        except Exception as e:
            if not args.quiet:
                print(f"Note: could not load Optuna ({e}); using embedded params.", file=sys.stderr)

    fmt = format_train_float_argv(params)
    bb, ig_line, ema_line = banner_lines_after_parse(
        fmt, int(params["ema_start_epoch"])
    )

    ok = (
        bb == GOLD_BANNER_LR_BACKBONE
        and ig_line == GOLD_BANNER_LR_INFOGATE
        and ema_line == GOLD_EMA_LINE
    )
    if not args.quiet:
        print("objective()-style argv snippets:")
        for k in (
            "learning_rate",
            "ig_learning_rate",
            "beta_ib",
            "mse_weight",
            "dropout_prob",
            "alpha_ib",
            "warmup_proportion",
            "weight_decay",
            "ema_decay",
            "selector_target_temp",
            "selector_rib_weight",
            "align_mix_floor",
            "gumbel_tau_start",
            "gumbel_tau_end",
        ):
            print(f"  --{k} {fmt[k]}")
        print()
        print("Expected train.py banner:")
        print(bb)
        print(ig_line)
        print(ema_line)
        print()
        if ok:
            print("OK: matches gold simsv2_phase4_trial_52.log header.")
        else:
            print("MISMATCH vs gold:")
            print(f"  want: {GOLD_BANNER_LR_BACKBONE}")
            print(f"  got:  {bb}")
            print(f"  want: {GOLD_BANNER_LR_INFOGATE}")
            print(f"  got:  {ig_line}")
            print(f"  want: {GOLD_EMA_LINE}")
            print(f"  got:  {ema_line}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
