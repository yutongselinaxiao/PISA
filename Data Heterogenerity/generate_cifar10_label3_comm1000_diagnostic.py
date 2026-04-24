"""Scenario-1 diagnostic: does comm_round=500 undercut sigma-regret
convergence on cifar10 label3?

Observation (dashboard section 6, 2026-04-23):
  cifar10 label3 Lipschitz textbook_sc has sigma/seed ratio ~1.49 -- the
  cross-sigma accuracy variation is *larger* than within-sigma seed noise,
  the only cell where this happens. The theorem (regret = O(log T / mu))
  predicts this ratio should shrink below 1 as T grows.

  Two candidate explanations:
    (1) comm_round=500 isn't enough time for sigma-regret to converge.
    (2) effective strong-convexity mu is smaller than assumed on this
        cell, so the same T gives a larger residual regret.

This script tests (1) by doubling comm_round to 1000. To keep the
parameter-free-gap comparison honest, BOTH the Lipschitz textbook_sc
runs AND the SISA original baseline are re-run at T=1000. Same seeds
(0, 1, 2) and same sigma0 sweep (1e2, 1e3, 1e4) as the T=500 runs in
paper-lipschitz-estimator.

Decision rules after runs finish:
  - If Lipschitz sigma/seed drops below 1 at T=1000 -> scenario 1
    confirmed, just run more rounds going forward.
  - If it stays >= 1 -> scenario 2 is more likely; revisit mu or
    sigma_max rather than adding rounds.
  - Re-run baseline at T=1000 so parameter_free_gap compares
    Lipschitz@1000 vs Original@1000, not Lipschitz@1000 vs Original@500.

Runs are tagged with `_comm1000` in the wandb run-name and local script
name so they are filterable alongside the existing T=500 runs.

Total: 3 sigma x 3 seeds x 2 methods = 18 runs on 8 GPUs.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_cifar10_label3_comm1000_diagnostic")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 2  # 18 runs; 8 x 2 = 16 workers gives ~1 wave.

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]
COMM_ROUND = "1000"
ETA_U_DECAY = "textbook_sc"

ONLINE_ENTRY = "experiment_sisa_practise_online.py"
ORIGINAL_ENTRY = "experiment_sisa_practise_wandb.py"

CASE = {
    "case_name": "cifar10_label3_n10",
    "dataset": "cifar10",
    "partition": "noniid-#label3",
    "model": "simple-cnn",
}

COMMON_ARGS = {
    "alg": "sisa",
    "lr": "0.001",
    "batch-size": "64",
    "epochs": "1",
    "n_parties": "10",
    "mu": "0.01",
    "rho": "0.9",
    "comm_round": COMM_ROUND,
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

LIPSCHITZ_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal_lipschitz",
    "sigma_min": "1e-6",
    "sigma_max": "1e6",
    "eta_u": "0.05",
    "eta_u_decay": ETA_U_DECAY,
    "G_clip": "5.0",
    "lipschitz_estimator": "ema",
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
}

# Two methods so the comparison at T=1000 is apples-to-apples.
JOB_SPECS = [
    {
        "spec_id": "lipschitz",
        "entry": ONLINE_ENTRY,
        "extra_args": LIPSCHITZ_EXTRA_ARGS,
        "tag": lambda slr: f"lipschitz_decay{ETA_U_DECAY}_sig{slr}_comm1000",
    },
    {
        "spec_id": "original",
        "entry": ORIGINAL_ENTRY,
        "extra_args": {},
        "tag": lambda slr: f"original_sig{slr}_comm1000",
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


def build_command(spec: dict, tag: str, cuda_device: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": CASE["model"],
        "dataset": CASE["dataset"],
        "partition": CASE["partition"],
    })
    args.update(spec["extra_args"])
    args["wandb_group"] = f"{CASE['case_name']}-{tag}"
    args["wandb_run_name"] = f"{CASE['dataset']}_{tag}_seed${{seed}}"

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {spec['entry']} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(spec: dict, sigma_lr: str, seed: int, tag: str,
                      cuda_device: str) -> str:
    cmd = build_command(spec, tag=tag, cuda_device=cuda_device)
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
        for slr in SIGMA_LR_VALUES:
            for seed in SEEDS:
                tag = spec["tag"](slr)
                script_name = f"{CASE['case_name']}_{tag}_seed{seed}.sh"
                jobs.append((spec, slr, seed, tag, script_name))

    generated_scripts = []
    for idx, (spec, slr, seed, tag, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            spec, sigma_lr=slr, seed=seed, tag=tag, cuda_device=gpu
        )
        script_path.write_text(script_text, encoding="utf-8")
        make_executable(script_path)
        generated_scripts.append((script_path, gpu, spec["spec_id"]))
        print(f"Generated: {script_path}  [GPU {gpu}]  ({spec['spec_id']})")

    total = len(generated_scripts)
    by_spec = {}
    for _, _, sid in generated_scripts:
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
        futs = [ex.submit(run_one, sp, gpu) for sp, gpu, _ in generated_scripts]
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
