"""Canonical FL σ-rule comparison sweep.

Four methods, same 9 cells (mnist/fmnist/cifar10 × label1/2/3), same σ_0
sweep ({1e2, 1e3, 1e4}), 3 seeds each. Total: 4 × 9 × 3 × 3 = 324 runs.

  1. original              → experiment_sisa_practise_wandb.py (paper SISA baseline)
  2. heuristic             → experiment_sisa_practise_online.py --sigma_mode heuristic
  3. convex_bal            → experiment_sisa_practise_online.py --sigma_mode online_convex_bal
  4. convex_bal_lipschitz  → experiment_sisa_practise_online.py --sigma_mode online_convex_bal_lipschitz

All `_online.py`-based modes use the canonical RMS residual aggregation
(fixed 2026-04-30 in experiment_sisa_practise_online.py near `avg_primal_res`).
Earlier `paper-lipschitz-estimator` runs used sum-of-norms aggregation; for a
clean comparison this sweep goes into a NEW project so the canonical and
non-canonical results don't get mixed.

Wandb project: paper-canonical-fl
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_canonical_fl")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 8

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

ONLINE_ENTRY = "experiment_sisa_practise_online.py"
ORIGINAL_ENTRY = "experiment_sisa_practise_wandb.py"

CASES = [
    {"case_name": "mnist_label1_n10",   "dataset": "mnist",   "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "mnist_label2_n10",   "dataset": "mnist",   "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "mnist_label3_n10",   "dataset": "mnist",   "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "fmnist_label1_n10",  "dataset": "fmnist",  "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "fmnist_label2_n10",  "dataset": "fmnist",  "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "fmnist_label3_n10",  "dataset": "fmnist",  "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "cifar10_label1_n10", "dataset": "cifar10", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "cifar10_label2_n10", "dataset": "cifar10", "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "cifar10_label3_n10", "dataset": "cifar10", "partition": "noniid-#label3", "model": "simple-cnn"},
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
    "datadir": "/data/yutong/datasets",
    "logdir": "./logs/",
    "noise": "0",
    "sample": "1",
    "sigma_lr": "${sigma_lr}",
    "rho_lr": "1e2",
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "paper-canonical-fl",
}

# Only the OGD modes need these knobs; original / heuristic ignore them.
OGD_BASE = {
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "eta_u": "0.05",
    "eta_u_decay": "textbook_sc",
    "G_clip": "5.0",
}

LIPSCHITZ_FLOOR = {
    "lipschitz_estimator": "ema",
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
    "lipschitz_floor_alpha": "1.0",
}

JOB_SPECS = [
    {
        "spec_id": "original",
        "entry": ORIGINAL_ENTRY,
        "extra_args": {},
        "tag": lambda sigma_lr: f"original_sig{sigma_lr}",
    },
    {
        "spec_id": "heuristic",
        "entry": ONLINE_ENTRY,
        "extra_args": {
            "sigma_mode": "heuristic",
            "sigma_min": "1e-6",
            "sigma_max": "1e8",
            "sigma_mu": "10.0",
            "sigma_tau": "2.0",
        },
        "tag": lambda sigma_lr: f"heuristic_mu10_tau2_sig{sigma_lr}",
    },
    {
        "spec_id": "convex_bal",
        "entry": ONLINE_ENTRY,
        "extra_args": {**OGD_BASE, "sigma_mode": "online_convex_bal"},
        "tag": lambda sigma_lr: f"convex_bal_sig{sigma_lr}",
    },
    {
        "spec_id": "convex_bal_lipschitz",
        "entry": ONLINE_ENTRY,
        "extra_args": {**OGD_BASE, **LIPSCHITZ_FLOOR,
                       "sigma_mode": "online_convex_bal_lipschitz"},
        "tag": lambda sigma_lr: f"convex_bal_lipschitz_sig{sigma_lr}",
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


def build_script_text(spec, case, sigma_lr, seed, tag, cuda_device):
    cmd = build_command(spec, case, tag=tag, cuda_device=cuda_device)
    return "\n".join([
        "#!/bin/bash", "", "set -e", "",
        f"sigma_lr={sigma_lr}", f"seed={seed}", "",
        cmd, "",
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
        script_path.write_text(build_script_text(spec, case, slr, seed, tag, gpu),
                               encoding="utf-8")
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
          f"per GPU ({max_workers} workers)...\n")
    gpu_sems = {g: threading.Semaphore(MAX_PARALLEL_PER_GPU) for g in CUDA_DEVICES}
    print_lock = threading.Lock()

    def run_one(script_path, gpu):
        log_path = LOG_DIR / f"{script_path.stem}.log"
        with gpu_sems[gpu]:
            with print_lock:
                print(f"Launching: {script_path.name} [GPU {gpu}] -> {log_path}")
            with open(log_path, "w") as log_file:
                p = subprocess.Popen(["bash", str(script_path)],
                                     stdout=log_file, stderr=subprocess.STDOUT)
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
        print("\nFailed:")
        for path, code, lp in failed:
            print(f"  {path} (exit {code}) -> {lp}")
    else:
        print("\nAll scripts completed successfully.")


if __name__ == "__main__":
    main()
