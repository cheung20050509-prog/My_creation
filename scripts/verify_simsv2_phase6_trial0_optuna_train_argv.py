#!/usr/bin/env python3
"""Verify SIMSv2 phase6 trial-0 argv formatting matches ``optuna_search_v2.objective()`` and gold log banner.

Gold train log header (``simsv2_phase6_simsv2_trial_0.log``):

  LR (backbone)  : 2.019142e-05
  LR (InfoGate)  : 0.0003523834

EMA line:

  EMA: decay=0.9994724074207356, start_epoch=3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fixed_experiment"))

from simsv2_phase6_trial0_hparams import (  # noqa: E402
    TRIAL_0_PARAMS,
    format_train_float_argv,
)


GOLD_BANNER_LR_BACKBONE = "  LR (backbone)  : 2.019142e-05"
GOLD_BANNER_LR_INFOGATE = "  LR (InfoGate)  : 0.0003523834"
GOLD_EMA_LINE = "EMA: decay=0.9994724074207356, start_epoch=3"


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
        help="Optional sqlite URI; if trial 0 exists, merge params from DB.",
    )
    ap.add_argument(
        "--study-name",
        default="infogate_simsv2_phase6_simsv2_4090d_space3",
        help="Study name for --storage.",
    )
    args = ap.parse_args()

    params = dict(TRIAL_0_PARAMS)
    if args.storage:
        try:
            import optuna  # type: ignore

            study = optuna.load_study(study_name=args.study_name, storage=args.storage)
            t = next(
                (x for x in study.get_trials(deepcopy=False) if x.number == 0), None
            )
            if t is None or not t.params:
                if not args.quiet:
                    print(
                        "Note: trial 0 not in DB; using embedded TRIAL_0_PARAMS.",
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
                if "seed" not in dbp:
                    dbp["seed"] = params["seed"]
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
                        "Loaded trial 0 params from Optuna storage "
                        f"(batch_config={bc})."
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
            print("OK: matches gold simsv2_phase6_simsv2_trial_0.log header.")
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
