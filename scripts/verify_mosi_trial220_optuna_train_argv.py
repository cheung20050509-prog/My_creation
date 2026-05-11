#!/usr/bin/env python3
"""Verify trial-220 hyperparameters use the same CLI float formatting as optuna_search_v2.objective().

Gold train log header (mosi_phase4_mosi_trial_220.log) shows parsed floats after argparse:
  LR (backbone)  : 2.927377e-05
  LR (InfoGate)  : 0.000245006

Raw Optuna/SQLite params use higher precision; objective() formats argv before subprocess.
"""
from __future__ import annotations

import argparse
import sys

# Params from phase4_mosi/run/mosi.log ``Trial 220 finished`` line (excluding batch_config).
TRIAL_220_PARAMS = {
    "n_epochs": 97,
    "learning_rate": 2.9273773228404854e-05,
    "ig_learning_rate": 0.00024500602832556887,
    "beta_ib": 22.973655873057357,
    "num_infogate_layers": 3,
    "bottleneck_dim": 192,
    "mse_weight": 1.2159459857149533,
    "dropout_prob": 0.24995585840082163,
    "alpha_ib": 0.0038406547193778717,
    "stage1_epochs": 6,
    "warmup_proportion": 0.10119382704338378,
    "weight_decay": 0.0007373939875592053,
    "ema_decay": 0.9951883680038079,
    "selector_target_temp": 0.7113722119040652,
    "selector_rib_weight": 0.060310067628194186,
    "gumbel_tau_start": 1.487505343680365,
    "gumbel_tau_end": 0.17652232232384735,
    "num_heads": 8,
    "unified_dim": 128,
    "ema_start_epoch": 5,
    "seed": 128,
}

GOLD_BANNER_LR_BACKBONE = "  LR (backbone)  : 2.927377e-05"
GOLD_BANNER_LR_INFOGATE = "  LR (InfoGate)  : 0.000245006"


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


def banner_lines_after_parse(fmt: dict[str, str]) -> tuple[str, str]:
    lr = float(fmt["learning_rate"])
    ig = float(fmt["ig_learning_rate"])
    return (
        f"  LR (backbone)  : {lr}",
        f"  LR (InfoGate)  : {ig}",
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
        help="Optional sqlite URI; if trial 220 exists, compare DB params to embedded snapshot.",
    )
    ap.add_argument(
        "--study-name",
        default="infogate_mosi_phase4_mosi_4090d",
        help="Study name for --storage.",
    )
    args = ap.parse_args()

    params = dict(TRIAL_220_PARAMS)
    if args.storage:
        try:
            import optuna  # type: ignore

            study = optuna.load_study(study_name=args.study_name, storage=args.storage)
            t = next(
                (x for x in study.get_trials(deepcopy=False) if x.number == 220), None
            )
            if t is None or not t.params:
                if not args.quiet:
                    print(
                        "Note: trial 220 not in DB; using embedded TRIAL_220_PARAMS.",
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
                        "Loaded trial 220 params from Optuna storage "
                        f"(batch_config={bc} skipped for float format check)."
                    )
        except Exception as e:
            if not args.quiet:
                print(f"Note: could not load Optuna ({e}); using embedded params.", file=sys.stderr)

    fmt = format_train_float_argv(params)
    bb, ig = banner_lines_after_parse(fmt)

    ok = bb == GOLD_BANNER_LR_BACKBONE and ig == GOLD_BANNER_LR_INFOGATE
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
        print("Expected train.py banner (first two LR lines):")
        print(bb)
        print(ig)
        print()
        if ok:
            print("OK: matches gold mosi_phase4_mosi_trial_220.log header.")
        else:
            print("MISMATCH vs gold:")
            print(f"  want: {GOLD_BANNER_LR_BACKBONE}")
            print(f"  got:  {bb}")
            print(f"  want: {GOLD_BANNER_LR_INFOGATE}")
            print(f"  got:  {ig}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
