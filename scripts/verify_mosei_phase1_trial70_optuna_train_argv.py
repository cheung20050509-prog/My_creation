#!/usr/bin/env python3
"""Verify MOSEI phase1 trial-70 hyperparameters match ``optuna_search_v2.objective()`` formatting.

Gold train log header (mosei_phase1_trial_70.log):
  LR (backbone)  : 3.078835e-05
  LR (InfoGate)  : 0.0005115383
  EMA: decay=0.999, start_epoch=5
"""

from __future__ import annotations

import argparse
import sys

# Tier 1 from phase1/run/mosei.log ``Trial 70 finished`` + DEFAULTS for unsampled knobs.
TRIAL_70_PARAMS = {
    "n_epochs": 50,
    "learning_rate": 3.078834774012053e-05,
    "ig_learning_rate": 0.0005115382788183823,
    "beta_ib": 21.749144451134427,
    "num_infogate_layers": 3,
    "bottleneck_dim": 64,
    "mse_weight": 2.145509477945063,
    "dropout_prob": 0.2680557435615756,
    "alpha_ib": 0.01,
    "stage1_epochs": 10,
    "warmup_proportion": 0.1,
    "weight_decay": 1e-3,
    "ema_decay": 0.999,
    "selector_target_temp": 0.35,
    "selector_rib_weight": 0.05,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.5,
    "num_heads": 4,
    "unified_dim": 256,
    "ema_start_epoch": 5,
    "seed": 128,
}

GOLD_BANNER_LR_BACKBONE = "  LR (backbone)  : 3.078835e-05"
GOLD_BANNER_LR_INFOGATE = "  LR (InfoGate)  : 0.0005115383"
GOLD_EMA_LINE = "EMA: decay=0.999, start_epoch=5"


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
        default=None,
        help="Optional sqlite URI; if trial 70 exists, merge tier1 params from DB.",
    )
    ap.add_argument(
        "--study-name",
        default="infogate_mosei_phase1_4090d",
        help="Study name for --storage.",
    )
    args = ap.parse_args()

    params = dict(TRIAL_70_PARAMS)
    if args.storage:
        try:
            import optuna  # type: ignore

            study = optuna.load_study(study_name=args.study_name, storage=args.storage)
            t = next(
                (x for x in study.get_trials(deepcopy=False) if x.number == 70), None
            )
            if t is None or not t.params:
                if not args.quiet:
                    print(
                        "Note: trial 70 not in DB; using embedded TRIAL_70_PARAMS.",
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
                        "Loaded trial 70 tier1 params from Optuna storage "
                        f"(batch_config={bc}); DEFAULTS unchanged."
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
            print("OK: matches gold mosei_phase1_trial_70.log header.")
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
