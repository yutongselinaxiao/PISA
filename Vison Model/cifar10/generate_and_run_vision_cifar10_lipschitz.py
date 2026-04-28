"""Vision-model CIFAR-10 + ResNet-34 sweep for the Lipschitz-floored online σ
adaptation method.

Mirrors the centralized-vision regime from the SISA paper's Table 3 (the cell
that produced the 95.04% ResNet-34 number) and adds our adaptive σ as an
alternative to the original lr-coupled σ schedule.

Three methods per cell, all writing to `paper-lipschitz-vision-cifar10`:
  - Original PISA fixed-σ schedule         (--sigma_mode fixed)
  - OGD on σ, no Lipschitz floor           (--sigma_mode online_convex_bal,
                                             bounded only by [sigma_min, sigma_max])
  - OGD on σ + BB Lipschitz floor          (--sigma_mode online_convex_bal_lipschitz)

σ-robustness sweep: σ_0 ∈ {0.1, 1e2, 1e3, 1e4} × 3 seeds × 3 methods = 36 runs.
σ_0 = 0.1 is the paper's tuned value (anchor / sanity check); 1e2-1e4 probe how
each method degrades when σ_0 is far from the tuned value. The middle method
(no-floor OGD) isolates the contribution of the BB Lipschitz floor itself
(present in method 3, absent in method 2).

Hyperparameters mirror PISA/Vison Model/cifar10/readme.md (the paper's
recommended ResNet-34 command). Differences from FL Data-Heterogenerity sweeps:
  - Centralized split (--num_gpu sub-batches per mini-batch, no client partition)
  - Total 205 epochs (paper) instead of comm_round=500
  - bs=128 (paper) instead of 64
  - σ-update fires once per epoch (sigma_update_freq = 50000/128 = 391)
    matching the FL convention of one σ-OGD step per round.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_vision_cifar10_lipschitz")
LOG_DIR = OUTPUT_DIR / "logs"

# ResNet-34 + bs=128 is heavy; 1 run/GPU is the safe default. Bump on bigger GPUs.
CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 1

SEEDS = [0, 1, 2]
# 0.1 is the paper's tuned value (sanity-check anchor); 1e2/1e3/1e4 probe σ₀-robustness.
SIGMA_LR_VALUES = ["0.1", "1e2", "1e3", "1e4"]

ESTIMATOR = "ema"
LIPSCHITZ_WINDOW_SIZE = "20"
ETA_U_DECAY = "textbook_sc"

# CIFAR-10 trainset size / batchsize = 50000 / 128 = ~391 batches per epoch.
# Fire one σ-OGD step per epoch to match the FL convention (~205 updates over
# the full run, comparable budget to comm_round=500 in the FL sweeps).
SIGMA_UPDATE_FREQ = "391"

ENTRY = "lower_loss_mPiAM_training_procedure_lipschitz.py"

# Single cell: ResNet-34 on cifar10 (matches paper's Table 3 headline).
# Add VGG-11 / DenseNet-121 cells later if needed -- their hyperparameters
# in readme.md differ enough that they warrant separate generators.
CASES = [
    {"case_name": "cifar10_resnet34", "model": "resnet"},
]

# Mirror PISA/Vison Model/cifar10/readme.md ResNet command.
COMMON_ARGS = {
    "optim": "adamw",
    "eps": "1e-8",
    "rho_lr": "5e3",
    "beta1": "0.9",
    "beta2": "0.999",
    "momentum": "0.9",
    "batchsize": "128",
    "total_epoch": "205",
    "decay_epoch": "3",
    "lr-gamma": "0.85",
    "weight_decay": "5e-5",
    "baseline_acc": "95.00",
    "beta_rmsprop": "0.999",
    # ours -- fixed across the sweep
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "G_clip": "5.0",
    "sigma_update_freq": SIGMA_UPDATE_FREQ,
    "lipschitz_estimator": ESTIMATOR,
    "lipschitz_window_size": LIPSCHITZ_WINDOW_SIZE,
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e10",
    "device": "cuda:0",
    # plumbed via shell template substitution
    "sigma_lr": "${sigma_lr}",
    "seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "paper-lipschitz-vision-cifar10",
}

LIPSCHITZ_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal_lipschitz",
    "eta_u": "0.05",
    "eta_u_decay": ETA_U_DECAY,
}

CONVEX_BAL_NO_FLOOR_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal",
    "eta_u": "0.05",
    "eta_u_decay": ETA_U_DECAY,
}

ORIGINAL_EXTRA_ARGS = {
    "sigma_mode": "fixed",
}

JOB_SPECS = [
    {
        "spec_id": "lipschitz_textbook_sc",
        "extra_args": LIPSCHITZ_EXTRA_ARGS,
        "tag": lambda sigma_lr: f"lipschitz_decay{ETA_U_DECAY}_sig{sigma_lr}",
    },
    {
        "spec_id": "convex_bal_no_floor",
        "extra_args": CONVEX_BAL_NO_FLOOR_EXTRA_ARGS,
        "tag": lambda sigma_lr: f"convex_bal_no_floor_decay{ETA_U_DECAY}_sig{sigma_lr}",
    },
    {
        "spec_id": "original",
        "extra_args": ORIGINAL_EXTRA_ARGS,
        "tag": lambda sigma_lr: f"original_sig{sigma_lr}",
    },
]

RUN_AFTER_GENERATION = True


def format_arg(key: str, value: str) -> str:
    val = str(value)
    if "${" in val:
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'--{key}="{escaped}"'
    escaped = val.replace("'", "'\"'\"'")
    return f"--{key}='{escaped}'"


def make_executable(path: Path):
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_wandb_names(case: dict, tag: str):
    group = f"{case['case_name']}-{tag}"
    run_name = f"{case['case_name']}_{tag}_seed${{seed}}"
    return group, run_name


def build_command(spec: dict, case: dict, tag: str, cuda_device: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({"model": case["model"]})
    args.update(spec["extra_args"])

    wandb_group, wandb_run_name = build_wandb_names(case, tag)
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {ENTRY} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(spec: dict, case: dict, sigma_lr: str, seed: int,
                      tag: str, cuda_device: str) -> str:
    cmd = build_command(spec, case, tag=tag, cuda_device=cuda_device)
    return "\n".join([
        "#!/bin/bash",
        "",
        "set -e",
        "",
        f"sigma_lr={sigma_lr}",
        f"seed={seed}",
        "",
        cmd,
        "",
    ])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in JOB_SPECS:
        for case in CASES:
            for slr in SIGMA_LR_VALUES:
                for seed in SEEDS:
                    tag = spec["tag"](slr)
                    script_name = f"{case['case_name']}_{tag}_seed{seed}.sh"
                    jobs.append((spec, case, slr, seed, tag, script_name))

    generated = []
    for idx, (spec, case, slr, seed, tag, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            spec, case, sigma_lr=slr, seed=seed, tag=tag, cuda_device=gpu
        )
        script_path.write_text(script_text, encoding="utf-8")
        make_executable(script_path)
        generated.append((script_path, gpu, spec["spec_id"]))
        print(f"Generated: {script_path}  [GPU {gpu}]  ({spec['spec_id']})")

    total = len(generated)
    by_spec = {}
    for _, _, sid in generated:
        by_spec[sid] = by_spec.get(sid, 0) + 1
    print(f"\nGenerated {total} single-seed scripts.")
    for sid, n in by_spec.items():
        print(f"  {sid}: {n}")

    if not RUN_AFTER_GENERATION:
        print("Not executing scripts.")
        return

    max_workers = len(CUDA_DEVICES) * MAX_PARALLEL_PER_GPU
    print(f"\nLaunching across GPUs {CUDA_DEVICES} with {MAX_PARALLEL_PER_GPU} "
          f"concurrent runs per GPU ({max_workers} workers total)...\n")

    gpu_sems = {g: threading.Semaphore(MAX_PARALLEL_PER_GPU) for g in CUDA_DEVICES}
    print_lock = threading.Lock()

    def run_one(script_path: Path, gpu: str):
        log_path = LOG_DIR / f"{script_path.stem}.log"
        with gpu_sems[gpu]:
            with print_lock:
                print(f"Launching: {script_path.name} [GPU {gpu}] -> {log_path}")
            with open(log_path, "w") as log_file:
                p = subprocess.Popen(
                    ["bash", str(script_path)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                ret = p.wait()
        return script_path, log_path, ret

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, sp, gpu) for sp, gpu, _ in generated]
        for fut in as_completed(futs):
            script_path, log_path, ret = fut.result()
            done += 1
            with print_lock:
                if ret == 0:
                    print(f"[{done}/{total}] Finished: {script_path.name}")
                else:
                    print(f"[{done}/{total}] FAILED: {script_path.name} "
                          f"(exit {ret}) -> {log_path}")
                    failed.append((script_path, ret, log_path))

    print("\nExecution finished.")
    if failed:
        print("\nFailed scripts:")
        for path, code, log_path in failed:
            print(f"  {path} (exit code {code}) -> {log_path}")
    else:
        print("\nAll scripts completed successfully.")


if __name__ == "__main__":
    main()
