"""Heuristic σ-update FL sweep.

Boyd's classical residual-balance multiplicative rule
  σ ← σ·τ if r > μ·s
  σ ← σ/τ if s > μ·r
  σ unchanged otherwise
on the same 9 FL cells (mnist/fmnist/cifar10 × label1/2/3) and σ_0 sweep
({1e2, 1e3, 1e4}) as the existing OGD / OGD+Lipschitz runs in
`paper-lipschitz-estimator`. Runs into the SAME wandb project so
build_results_dashboard.py can aggregate all four σ-update methods
(fixed / heuristic / online_convex_bal / online_convex_bal_lipschitz)
side-by-side.

Implementation: experiment_sisa_practise_online.py with --sigma_mode=heuristic.
The heuristic update is at experiment_sisa_practise_online.py:62 and dispatched
at experiment_sisa_practise_online.py:1801. The same script handles
fixed / OGD modes too.

Total: 3 datasets x 3 partitions x 3 σ_0 x 3 seeds = 81 runs.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_heuristic")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 8

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

# Boyd's heuristic parameters. μ=10, τ=2 are the canonical values from the
# ADMM textbook (Boyd 2011, Distributed Optimization and Statistical Learning).
HEURISTIC_MU = "10.0"
HEURISTIC_TAU = "2.0"

ONLINE_ENTRY = "experiment_sisa_practise_online.py"

# Same 9 cells as the existing Lipschitz-floor sweep.
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

# Common args mirror the existing Lipschitz-floor sweep so the heuristic
# rows are apples-to-apples with the OGD rows in the dashboard.
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
    "wandb_project": "paper-lipschitz-estimator",
}

HEURISTIC_EXTRA_ARGS = {
    "sigma_mode": "heuristic",
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "sigma_mu": HEURISTIC_MU,    # heuristic threshold (note: parser uses --sigma_mu)
    "sigma_tau": HEURISTIC_TAU,  # heuristic multiplier (parser: --sigma_tau)
}

JOB_SPEC = {
    "spec_id": "heuristic",
    "entry": ONLINE_ENTRY,
    "extra_args": HEURISTIC_EXTRA_ARGS,
    "cases": CASES,
    "seeds": SEEDS,
    "tag": lambda sigma_lr: f"heuristic_mu10_tau2_sig{sigma_lr}",
}

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
    for case in JOB_SPEC["cases"]:
        for slr in SIGMA_LR_VALUES:
            for seed in JOB_SPEC["seeds"]:
                tag = JOB_SPEC["tag"](slr)
                script_name = f"{case['case_name']}_{tag}_seed{seed}.sh"
                jobs.append((JOB_SPEC, case, slr, seed, tag, script_name))

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
    print(f"\nGenerated {total} single-seed scripts (heuristic only).")

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
