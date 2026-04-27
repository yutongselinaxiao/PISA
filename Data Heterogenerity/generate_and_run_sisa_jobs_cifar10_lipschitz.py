"""CIFAR-10 + ResNet50 sweep for the Lipschitz-floor (online_convex_bal_lipschitz +
textbook_sc) method.

Adds the standard CV-FL benchmark to the paper's main results table alongside
mnist/fmnist (in `paper-lipschitz-estimator`) and femnist (separate generator).

Two methods per cell, both writing to `paper-lipschitz-estimator-cifar-resnet`:
  - Lipschitz textbook_sc adaptive sigma (experiment_sisa_practise_online.py)
  - Original SISA fixed-sigma baseline (experiment_sisa_practise_wandb.py)

Cases:
  - partition="noniid-#label{1,2,3}" -- matches mnist/fmnist/femnist label
    convention. K=10 default in utils.py covers all 10 cifar10 classes.

Concurrency note: ResNet50 on cifar10 is much heavier than simple-cnn on
mnist-family. MAX_PARALLEL_PER_GPU is dropped from 8 to 2 to avoid OOM.
With bs=64 a single ResNet50 run on cifar10 typically uses ~5-7 GB; 2/GPU
fits comfortably on 16+ GB cards. Bump up if you have 24-40 GB cards and
want to saturate.

Total: 3 partitions x 3 sigma x 3 seeds x 2 methods = 54 runs at T=500.
Wall time: ~5-10x FEMNIST per run (ResNet50 vs simple-cnn).
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_cifar10_lipschitz")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 2  # ResNet50 is heavy; bump up if you have 24+ GB cards

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

ESTIMATOR = "ema"
LIPSCHITZ_WINDOW_SIZE = "20"
ETA_U_DECAY = "textbook_sc"  # parameter-free strongly-convex schedule

ONLINE_ENTRY = "experiment_sisa_practise_online.py"
ORIGINAL_ENTRY = "experiment_sisa_practise_wandb.py"

# CIFAR-10 cases. K=10 default in utils.py covers all 10 classes for the
# noniid-#label{1,2,3} family. No "real" partition for cifar10 (that branch
# is femnist-only at utils.py:453).
CASES = [
    {"case_name": "cifar10_label1_n10", "dataset": "cifar10",
     "partition": "noniid-#label1", "model": "resnet"},
    {"case_name": "cifar10_label2_n10", "dataset": "cifar10",
     "partition": "noniid-#label2", "model": "resnet"},
    {"case_name": "cifar10_label3_n10", "dataset": "cifar10",
     "partition": "noniid-#label3", "model": "resnet"},
]

COMMON_ARGS = {
    "alg": "sisa",
    "lr": "0.001",
    "batch-size": "64",
    "epochs": "1",
    "n_parties": "10",
    "mu": "0.01",
    "rho": "0.9",
    "comm_round": "500",
    "beta": "0.5",
    "device": "cuda:0",
    "datadir": "/dataMeR2/yutong/datasets",
    "logdir": "./logs/",
    "noise": "0",
    "sample": "1",
    "sigma_lr": "${sigma_lr}",
    "rho_lr": "1e2",
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "paper-lipschitz-estimator-cifar-resnet",
}

LIPSCHITZ_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal_lipschitz",
    "sigma_min": "1e-6",
    "sigma_max": "1e6",
    "eta_u": "0.05",
    "eta_u_decay": ETA_U_DECAY,
    "G_clip": "5.0",
    "lipschitz_estimator": ESTIMATOR,
    "lipschitz_window_size": LIPSCHITZ_WINDOW_SIZE,
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
}

# Two specs: Lipschitz textbook_sc (online entry) and SISA baseline (wandb entry).
JOB_SPECS = [
    {
        "spec_id": "lipschitz_textbook_sc",
        "entry": ONLINE_ENTRY,
        "extra_args": LIPSCHITZ_EXTRA_ARGS,
        "cases": CASES,
        "seeds": SEEDS,
        "tag": lambda sigma_lr: f"lipschitz_decay{ETA_U_DECAY}_sig{sigma_lr}",
    },
    {
        "spec_id": "original_sisa",
        "entry": ORIGINAL_ENTRY,
        "extra_args": {},
        "cases": CASES,
        "seeds": SEEDS,
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
    run_name = f"{case['dataset']}_{tag}_seed${{seed}}"
    return group, run_name


def build_command(spec: dict, case: dict, tag: str, cuda_device: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "partition": case["partition"],
    })
    args.update(spec["extra_args"])

    wandb_group, wandb_run_name = build_wandb_names(case, tag)
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {spec['entry']} \\"]
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
        for case in spec["cases"]:
            for slr in SIGMA_LR_VALUES:
                for seed in spec["seeds"]:
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
