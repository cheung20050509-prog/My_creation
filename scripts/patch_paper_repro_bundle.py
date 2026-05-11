#!/usr/bin/env python3
"""Apply bundle-only path hunks to ``paper_repro_bundle/code/train.py`` and ``deberta_infogate.py``.

Idempotent: skips files that already contain ``PAPER_REPRO_BUNDLE_MARKER``.
"""

from __future__ import annotations

import sys


MARKER = "PAPER_REPRO_BUNDLE_MARKER"


def patch_train(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        s = f.read()
    if MARKER in s:
        return

    anchor = "from selection_utils import (\n    DEFAULT_SELECTION_METRIC,\n    SELECTION_METRIC_CHOICES,\n    build_selection_tiebreak,\n    compute_selection_score,\n    selection_higher_is_better,\n)\n\n"
    if anchor not in s:
        raise SystemExit(f"patch_train: anchor not found in {path}")

    insert = (
        "from selection_utils import (\n"
        "    DEFAULT_SELECTION_METRIC,\n"
        "    SELECTION_METRIC_CHOICES,\n"
        "    build_selection_tiebreak,\n"
        "    compute_selection_score,\n"
        "    selection_higher_is_better,\n)\n\n"
        f"# {MARKER}: this file lives under paper_repro_bundle/code/\n"
        "_BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
        '_WEIGHTS_ROOT = os.path.join(_BUNDLE_ROOT, "weights")\n'
        '_DATASETS_ROOT = os.path.join(_BUNDLE_ROOT, "data", "datasets")\n\n'
    )
    s = s.replace(anchor, insert, 1)

    old_default = (
        'parser.add_argument("--model", type=str,\n'
        '                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "deberta-v3-base"))'
    )
    new_default = (
        'parser.add_argument("--model", type=str,\n'
        '                    default=os.path.join(_WEIGHTS_ROOT, "deberta-v3-base"))'
    )
    if old_default not in s:
        raise SystemExit(f"patch_train: --model default not found in {path}")
    s = s.replace(old_default, new_default, 1)

    old_sims = (
        'if args.dataset == "simsv2":\n'
        '    if "deberta-v3-base" in args.model:\n'
        '        args.model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert-base-chinese")\n'
    )
    new_sims = (
        'if args.dataset == "simsv2":\n'
        '    if "deberta-v3-base" in args.model:\n'
        '        args.model = os.path.join(_WEIGHTS_ROOT, "bert-base-chinese")\n'
    )
    if old_sims not in s:
        raise SystemExit(f"patch_train: simsv2 model swap not found in {path}")
    s = s.replace(old_sims, new_sims, 1)

    old_setup = '    with open(f"datasets/{args.dataset}.pkl", "rb") as fh:\n'
    new_setup = (
        '    with open(os.path.join(_DATASETS_ROOT, f"{args.dataset}.pkl"), "rb") as fh:\n'
    )
    if old_setup not in s:
        raise SystemExit(f"patch_train: setup_data open not found in {path}")
    s = s.replace(old_setup, new_setup, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def patch_deberta(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        s = f.read()
    if MARKER in s:
        return

    old = '_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deberta-v3-base")\n'
    new = (
        f"# {MARKER}\n"
        "_FE_DIR = os.path.dirname(os.path.abspath(__file__))\n"
        "_BUNDLE_ROOT = os.path.dirname(_FE_DIR)\n"
        '_MODEL_DIR = os.path.join(_BUNDLE_ROOT, "weights", "deberta-v3-base")\n'
    )
    if old not in s:
        raise SystemExit(f"patch_deberta: _MODEL_DIR line not found in {path}")
    s = s.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_paper_repro_bundle.py <paper_repro_bundle_root>", file=sys.stderr)
        return 2
    root = sys.argv[1].rstrip("/")
    patch_train(f"{root}/code/train.py")
    patch_deberta(f"{root}/code/deberta_infogate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
