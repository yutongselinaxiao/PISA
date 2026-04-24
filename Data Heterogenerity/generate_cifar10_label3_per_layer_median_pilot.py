"""Per-layer-median L̂ diagnostic pilot on cifar10 (all partitions).

Context (2026-04-24): At T=1000 on cifar10 label3, sigma_0 in {1e2, 1e3, 1e4}
converge to DIFFERENT final accuracies (0.148, 0.152, 0.214) — the one cell
where sigma-regret has not converged to the seed-noise floor. The per-layer
L̂ diagnostic showed max_over_median spikes up to 11.3 when sigma_0 is small,
suggesting the scalar floor is being dragged up by one outlier layer. Label1
and label2 on cifar10 have smaller but non-trivial per-layer heterogeneity
(max/median in the 2-12 range) and also showed negative parameter-free gaps
in the dashboard, so they are in scope too.

This pilot tests the cheap diagnostic: swap the scalar L̂ from global-norm
to the median of per-layer L̂ EMAs. It does NOT implement per-layer sigma;
it only softens the scalar floor. Two outcomes:
  - If sigma_0=1e2 accuracy jumps to match sigma_0=1e4 (both land near the
    current best): outlier layer was the bottleneck, full per-layer sigma
    will help. Proceed to build experiment_sisa_practise_online_perlayer.py.
  - If sigma_0=1e2 is still stuck low: per-layer sigma won't rescue it
    either; cifar10 is fundamentally hard for this formulation.

Added in online.py: new --lipschitz_estimator value `ema_per_layer_median`.
Pilot covers cifar10 x {label1, label2, label3} x sigma_0 in {1e2, 1e3, 1e4}
x seeds {0, 1, 2} at comm_round=500 for fast turn-around; if signal appears,
extend to 1000.

Total: 3 partitions x 3 sigma x 3 seeds = 27 runs on 8 GPUs.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_cifar10_label3_per_layer_median_pilot")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 2  # 9 runs total; plenty of headroom.

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]
COMM_ROUND = "500"
ETA_U_DECAY = "textbook_sc"
ESTIMATOR = "ema_per_layer_median"  # the new mode added in online.py

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
    "lipschitz_estimator": ESTIMATOR,
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
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


def make_tag(sigma_lr: str) -> str:
    # `_plmedian` tag keeps these filterable in the dashboard alongside
    # the existing scalar-L̂ runs.
    return f"lipschitz_decay{ETA_U_DECAY}_plmedian_sig{sigma_lr}"


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


def build_script_text(case: dict, sigma_lr: str, seed: int, tag: str,
                      cuda_device: str) -> str:
    cmd = build_command(case, tag=tag, cuda_device=cuda_device)
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
    for case in CASES:
        for slr in SIGMA_LR_VALUES:
            for seed in SEEDS:
                tag = make_tag(slr)
                script_name = f"{case['case_name']}_{tag}_seed{seed}.sh"
                jobs.append((case, slr, seed, tag, script_name))

    generated = []
    for idx, (case, slr, seed, tag, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            case, sigma_lr=slr, seed=seed, tag=tag, cuda_device=gpu
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
