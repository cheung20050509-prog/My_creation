#!/usr/bin/env python3
"""Resume MUStARD stage-2 local TPE on an existing two-stage classify run."""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

import optuna  # noqa: E402

from optuna_search_classify import (  # noqa: E402
    build_local_search_space,
    build_stage_db_uri,
    build_two_stage_study_names,
    clone_cli,
    create_study_for_cli,
    optimize_with_cleanup,
    print_study_header,
    print_study_summary,
    summarize_local_space,
    trial_ckpt_base,
)


def main():
    pa = argparse.ArgumentParser(description="Continue MUStARD stage-2 only")
    pa.add_argument("--run_root", type=str, required=True)
    pa.add_argument(
        "--base_study",
        type=str,
        required=True,
        help="e.g. infogate_mustard_optuna_classify_albert_hcf_20260426_192758",
    )
    pa.add_argument("--n_trials", type=int, default=200)
    pa.add_argument("--search_tier", type=int, default=2, choices=[1, 2, 3])
    pa.add_argument("--stage2_top_k", type=int, default=8)
    pa.add_argument("--n_startup_trials", type=int, default=20)
    pa.add_argument("--selection_metric", type=str, default="binary_acc")
    pa.add_argument("--gpu", type=int, default=0)
    args = pa.parse_args()

    run_root = os.path.abspath(args.run_root)
    base_db = f"sqlite:///{os.path.join(run_root, 'db', 'mustard.db')}"
    cli = argparse.Namespace(
        dataset="mustard",
        study_name=args.base_study,
        db=base_db,
        artefact_root=run_root,
        search_tier=args.search_tier,
        selection_metric=args.selection_metric,
        n_startup_trials=args.n_startup_trials,
        stage2_top_k=args.stage2_top_k,
        stage2_trials=args.n_trials,
        n_trials=args.n_trials,
        n_epochs=None,
        n_epochs_min=None,
        disable_two_stage=False,
        disable_l_lib=False,
        disable_l_rib=False,
        stage_label=None,
        enqueue_top_from=None,
        enqueue_top_k=10,
        gpu=args.gpu,
        stage1_trials=40,
    )

    s1_name, s2_name = build_two_stage_study_names(cli)
    s1_db = build_stage_db_uri(s1_name, base_db)
    stage1_study = optuna.load_study(study_name=s1_name, storage=s1_db)

    local_space, top_trials = build_local_search_space(
        stage1_study,
        cli.dataset,
        cli.search_tier,
        cli.selection_metric,
        cli.stage2_top_k,
    )
    top_ids = [t.number for t in top_trials]
    print("\n" + "=" * 60)
    print("MUStARD stage-2 continuation")
    print(f"  Run root: {run_root}")
    print(f"  Stage-1 study: {s1_name} ({len(stage1_study.trials)} trials)")
    print(f"  Anchors top-{len(top_ids)}: {top_ids}")
    print(summarize_local_space(local_space), end="")

    stage2_cli = clone_cli(
        cli,
        study_name=s2_name,
        db=build_stage_db_uri(s2_name, base_db),
        n_trials=cli.stage2_trials,
        sampler_name="tpe",
        sampler_seed=256,
        stage_label="s2_local",
        local_space=local_space,
    )
    stage2_study, stage2_mode = create_study_for_cli(stage2_cli)
    print_study_header(stage2_cli, stage2_mode, len(stage2_study.trials))
    optimize_with_cleanup(
        stage2_study,
        stage2_cli,
        trial_ckpt_base(stage2_cli, cli.dataset, "s2_local"),
    )
    print_study_summary(stage2_study, stage2_cli)


if __name__ == "__main__":
    main()
