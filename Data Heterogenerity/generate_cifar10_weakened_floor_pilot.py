"""Weakened-floor pilot on cifar10 (all partitions).

Context (2026-04-24): With the hard Lipschitz projection (alpha=1) the
cifar10 simple-cnn runs plateau at ~0.21 regardless of sigma_0, while
tuned SISA at sigma=1e3 reaches ~0.40 at T=1000. Trajectory analysis
showed the BB-estimated L_hat on cifar10 is ~10^4-10^5, 10x higher than
the productive sigma range (~10^3), so the hard floor over-regularizes
sigma. See online_convex_bal_lipschitz_update_u docstring (2026-04-24
CHANGE LOG) for the new `--lipschitz_floor_alpha` parameter.

This pilot tests alpha in {0.01, 0.1, 0.3} against the default alpha=1.0
(existing data) on cifar10 x {label1, label2, label3} x sigma_0 in
{1e2, 1e3, 1e4} x seeds {0, 1, 2}. If any alpha<1 brings the ceiling
up meaningfully (say >0.30 on label3), weakened floor is a viable fix.
If all alpha values plateau at ~0.21, cifar10 is a formulation-level
limitation and we stop trying to rescue it.

Total: 3 partitions x 3 alpha x 3 sigma x 3 seeds = 81 runs on 8 GPUs.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_cifar10_weakened_floor_pilot")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 4  # simple-cnn batch=64 is tiny; H100 80GB fits 4 easily

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]
ALPHA_VALUES = ["0.01", "0.1", "0.3"]  # alpha<1 relaxes floor
COMM_ROUND = "500"
ETA_U_DECAY = "textbook_sc"

ONLINE_ENTRY = "experiment_sisa_practise_online.py"

CASES = [
    {"case_name": "cifar10_label1_n10", "dataset": "cifar10",
     "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "cifar10_label2_n10", "dataset": "cifar10",
     "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "cifar10_label3_n10", "dataset": "cifar10",
     "partition": "noniid-#label3", "model": "simple-cnn"},
]

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
    "datadir": "/dataMeR2/yutong/datasets",
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

LIPSCHITZ_ARGS = {
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
    "lipschitz_floor_alpha": "${alpha}",
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


def make_tag(sigma_lr: str, alpha: str) -> str:
    alpha_safe = alpha.replace(".", "p")
    return f"lipschitz_decay{ETA_U_DECAY}_floor{alpha_safe}_sig{sigma_lr}"


def build_command(case: dict, tag: str, cuda_device: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "partition": case["partition"],
    })
    args.update(LIPSCHITZ_ARGS)
    args["wandb_group"] = f"{case['case_name']}-{tag}"
    args["wandb_run_name"] = f"{case['dataset']}_{tag}_seed${{seed}}"

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {ONLINE_ENTRY} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(case: dict, sigma_lr: str, alpha: str, seed: int,
                      tag: str, cuda_device: str) -> str:
    cmd = build_command(case, tag=tag, cuda_device=cuda_device)
    return "\n".join([
        "#!/bin/bash",
        "",
        "set -e",
        "",
        f"sigma_lr={sigma_lr}",
        f"alpha={alpha}",
        f"seed={seed}",
        "",
        cmd,
        "",
    ])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for case in CASES:
        for slr in SIGMA_LR_VALUES:
            for alpha in ALPHA_VALUES:
                for seed in SEEDS:
                    tag = make_tag(slr, alpha)
                    script_name = f"{case['case_name']}_{tag}_seed{seed}.sh"
                    jobs.append((case, slr, alpha, seed, tag, script_name))

    generated = []
    for idx, (case, slr, alpha, seed, tag, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            case, sigma_lr=slr, alpha=alpha, seed=seed, tag=tag, cuda_device=gpu
        )
        script_path.write_text(script_text, encoding="utf-8")
        make_executable(script_path)
        generated.append((script_path, gpu))
        print(f"Generated: {script_path}  [GPU {gpu}]")

    total = len(generated)
    print(f"\nGenerated {total} single-seed scripts.")

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
        futs = [ex.submit(run_one, sp, gpu) for sp, gpu in generated]
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
