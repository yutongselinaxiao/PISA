"""Canonical sigma-rule comparison for GPT-2 Medium and XL.

Three methods (matching `_online.py`):
  1. original      - LR-coupled fixed schedule, no residual feedback.
  2. ogd           - canonical OGD on (u - target)^2, target = log(primal/dual).
  3. ogd_lipschitz - canonical OGD + hard Lipschitz floor u >= log(alpha * L_hat).

Two model sizes (Medium / XL) with size-appropriate sigma_0 ranges:
  Medium (350M params, n_layer=24): sigma_0 in {8e1, 8e2, 8e3}  -- centered on
        the paper-tuned 8e2 anchor in train_gpt2_medium_sisa.py.
  XL    (1.5B params, n_layer=48): sigma_0 in {1e2, 1e3, 1e4}  -- shifted up
        because XL's loss surface is larger / smoother.

Seeds: 1337, 1338, 1339 (3 per cell).

Total cells: 3 methods x 2 sizes x 3 sigma_0 x 3 seeds = 54 runs.

Each run uses 4-8 GPUs via torchrun. Given each GPT-2 training takes hours
to days, this script does NOT auto-launch -- it emits the torchrun commands
to a shell file for manual review and serial / staged execution.

Wandb projects:
  Medium: gpt2-sisa-canonical
  XL:     gpt2-sisa-canonical-xl

NAMING NOTE (from project_ogd_admm_vs_sisa_residuals memory): the GPT-2
"ogd" mode now uses the SAME canonical OGD as `_online.py`'s
`online_convex_bal[_lipschitz]` modes -- target = log(primal/dual), no
LR-coupled anchor, no EMA on residuals. Run-name prefix `ogd_gpt` should
NOT be confused with the simple-cnn `ogd_admm` (residuals from
augmented-Lagrangian Adam/SGD local solve in
`experiment_sisa_practise_admm.py`) or `ogd_sisa` (residuals from the
closed-form SISA solve in `experiment_sisa_practise_online.py`). For the
GPT-2 runs in this sweep, the local solve is the
DistributedOptimizer.linearized SISA-style closed-form step, so the
residuals are most analogous to `ogd_sisa` -- but on a different model
class (GPT-2) and different scale.
"""

import argparse
import os
import stat
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "generated_canonical_gpt2"
LAUNCH_SCRIPT = OUTPUT_DIR / "launch_all.sh"

ENTRY = "train_gpt_sisa_lower_no_2ndgradient_online.py"

SIZES = {
    "medium": {
        "config_file": "config/train_gpt2_medium_{method}.py",
        "wandb_project": "gpt2-sisa-canonical",
        "sigma_lr_values": ["8e1", "8e2", "8e3"],
        "default_nproc": 4,
    },
    "xl": {
        "config_file": "config/train_gpt2_xl_{method}.py",
        "wandb_project": "gpt2-sisa-canonical-xl",
        "sigma_lr_values": ["1e2", "1e3", "1e4"],
        "default_nproc": 4,
    },
}

METHODS = [
    "original",                  # LR-coupled fixed schedule, no σ-rule
    "ogd",                       # canonical OGD on (u - log(primal/dual))²
    "ogd_lipschitz",             # canonical OGD + BB-Lipschitz hard projection
    "ogd_anchored_canonical",    # anchored OGD with Boyd-canonical residual
    "ogd_anchored_old",          # anchored OGD with OLD non-canonical residual
]
SEEDS = [1337, 1338, 1339]


def slug(value: str) -> str:
    return value.replace(".", "p").replace("+", "")


def build_command(size: str, method: str, sigma_lr: str, seed: int, nproc: int,
                  gpus: str | None):
    info = SIZES[size]
    config_path = info["config_file"].format(method=method)
    tag = f"gpt2_{size}_{method}_sig{slug(sigma_lr)}_seed{seed}"
    save_root = "/dataMeR2/yutong/sisa_gpt2"

    # NOTE: configurator.py expects --key=value (with -- prefix) for overrides;
    # bare names are interpreted as config-file paths. See configurator.py L20-23.
    # configurator.py also enforces type(attempt) == type(globals()[key]) (L42),
    # so we cannot override globals whose default is None (e.g. wandb_run_name)
    # with a string -- skip that override and rely on the train script's
    # fallback `name = wandb_run_name or comment` (line 1240).
    overrides = [
        f"--sigma_lr={sigma_lr}",
        f"--seed={seed}",
        f"--comment={tag}",
        f"--save_dir={save_root}/log_gpt2/{tag}",
        f"--out_dir={save_root}/out_gpt2/{tag}",
    ]

    prefix = f"CUDA_VISIBLE_DEVICES={gpus} " if gpus else ""
    cmd_parts = [
        f"{prefix}torchrun --standalone --nproc_per_node={nproc}",
        ENTRY,
        config_path,
    ] + overrides

    return tag, " \\\n    ".join(cmd_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc-per-node", type=int, default=None,
                    help="override default nproc_per_node for both sizes "
                         "(default: 4 for both medium and xl, matching the "
                         "current 4xH100 machine).")
    ap.add_argument("--gpus", type=str, default="0,1,2,3",
                    help="comma-separated GPU IDs for CUDA_VISIBLE_DEVICES "
                         "(default: 0,1,2,3). Pass an empty string to omit "
                         "the env-var prefix entirely.")
    ap.add_argument("--medium-only", action="store_true")
    ap.add_argument("--xl-only", action="store_true")
    ap.add_argument("--methods", type=str, default=None,
                    help="comma-separated subset of methods to run (e.g. "
                         "'ogd,ogd_lipschitz' to skip already-completed "
                         "'original'). Default: all of {original, ogd, ogd_lipschitz}.")
    args = ap.parse_args()
    methods = METHODS if args.methods is None else [
        m.strip() for m in args.methods.split(",") if m.strip()
    ]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Valid: {METHODS}")
    gpus = args.gpus.strip() or None
    # Sanity: warn if nproc and number of visible GPUs disagree.
    if gpus is not None:
        n_visible = len([g for g in gpus.split(",") if g.strip()])
        for size, info in SIZES.items():
            requested = args.nproc_per_node or info["default_nproc"]
            if requested != n_visible:
                print(f"WARNING: {size} uses nproc_per_node={requested} but "
                      f"--gpus exposes {n_visible} GPUs ({gpus}). torchrun "
                      "will likely fail or under-use GPUs.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sizes = list(SIZES.keys())
    if args.medium_only:
        sizes = ["medium"]
    elif args.xl_only:
        sizes = ["xl"]

    lines = ["#!/bin/bash", "", "set -e", "set -o pipefail", ""]
    n_jobs = 0
    for size in sizes:
        info = SIZES[size]
        nproc = args.nproc_per_node or info["default_nproc"]
        for method in methods:
            for sigma_lr in info["sigma_lr_values"]:
                for seed in SEEDS:
                    tag, cmd = build_command(size, method, sigma_lr, seed, nproc, gpus)
                    log_path = f"{OUTPUT_DIR.name}/logs/{tag}.log"
                    lines.append(f"# --- {tag} ---")
                    lines.append(f"mkdir -p $(dirname {log_path})")
                    lines.append(f"echo \"[$(date)] launching {tag}\"")
                    lines.append(f"({cmd}) 2>&1 | tee {log_path}")
                    lines.append("")
                    n_jobs += 1

    LAUNCH_SCRIPT.write_text("\n".join(lines), encoding="utf-8")
    mode = LAUNCH_SCRIPT.stat().st_mode
    LAUNCH_SCRIPT.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Wrote {LAUNCH_SCRIPT} ({n_jobs} runs, sequential).")
    print("\nTo run all:")
    print(f"  cd {Path(__file__).resolve().parent}")
    print(f"  ./{LAUNCH_SCRIPT.relative_to(Path(__file__).resolve().parent)}")
    print("\nTo run a subset, copy individual blocks from the script.")
    print("\nApprox wallclock per run (50000 iters):")
    print("  Medium @ 4 H100s : ~24-32 hours")
    print("  XL     @ 4 H100s : ~6-8 days")
    print("  Total (54 runs) : ~6-8 weeks if strictly serial; do staged.")
    print("  Suggested staging: Medium (27 runs) first; XL only after Medium")
    print("  results show which sigma_0 values are productive.")


if __name__ == "__main__":
    main()
