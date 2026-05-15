"""Spectral AADMM (Xu-Figueiredo-Goldstein 2017) baseline on the FL sweep.

Adds the spectral-adaptive ρ baseline to the σ_0-robustness comparison.
The math is BB-style: estimate two stepsizes from the deltas of the
intermediate / final dual variables, gated by a correlation safeguard.

Differences vs. our canonical OGD+Lipschitz:
  - No log-domain step (sigma_lr is updated in raw space).
  - Update only fires when cor(Δh, Δλ_hat) AND/OR cor(Δg, Δλ) > 0.2.
  - When safeguard rejects, sigma_lr is left unchanged (no fallback target).

The ports lives in `experiment_sisa_practise_online.py` (sigma_mode
'spectral_aadmm'); math is from controllers.py:SpectralAADMM, which is
otherwise inert to running scripts.

Sweep:
  - 9 cells (mnist/fmnist/cifar10 × label1/2/3)
  - σ_0 ∈ {1e2, 1e3, 1e4} (matches canonical-fl grid for direct comparison)
  - 3 seeds
  - Single method ('spectral_aadmm')
  - SGD ep=1 (canonical-fl default)

Total: 1 × 9 × 3 × 3 = 81 runs.

Wandb project: paper-canonical-fl-spectral-aadmm
  - Distinct from `paper-canonical-fl` so the new baseline is cleanly
    separable from the headline comparison.
"""
import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_spectral_aadmm")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3"]
MAX_PARALLEL_PER_GPU = 8

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

ENTRY = "experiment_sisa_practise_online.py"

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
    "datadir": "/dataMeR2/yutong/datasets",
    "logdir": "./logs/",
    "noise": "0",
    "sample": "1",
    "sigma_lr": "${sigma_lr}",
    "rho_lr": "1e2",
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "paper-canonical-fl-spectral-aadmm",
    # Spectral AADMM mode
    "sigma_mode": "spectral_aadmm",
    "sigma_min": "1e-6",  # not consumed by spectral path but argparse expects
    "sigma_max": "1e8",
    "sigma_update_freq": "1",
    # Spectral knobs (defaults match Xu-Figueiredo-Goldstein 2017)
    "spectral_update_period": "2",
    "spectral_correlation_threshold": "0.2",
    "spectral_sigma_min": "1e-6",
    "spectral_sigma_max": "1e8",
    # OGD / Lipschitz knobs are not consumed by spectral_aadmm but argparse
    # still expects them; keep defaults that match the floor_only / canonical
    # generators so all paper-canonical-fl-* projects use identical
    # baseline-irrelevant args.
    "eta_u": "0.05",
    "eta_u_decay": "textbook_sc",
    "G_clip": "5.0",
    "lipschitz_estimator": "ema",
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
    "lipschitz_floor_alpha": "1.0",
}

RUN_AFTER_GENERATION = True


def format_arg(key, value):
    val = str(value)
    if "${" in val:
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'--{key}="{escaped}"'
    escaped = val.replace("'", "'\"'\"'")
    return f"--{key}='{escaped}'"


def make_executable(path):
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_command(case, sigma_lr, cuda_device):
    args = dict(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "partition": case["partition"],
    })
    tag = f"spectral_aadmm_sig{sigma_lr}"
    args["wandb_group"] = f"{case['case_name']}-{tag}"
    args["wandb_run_name"] = f"{case['dataset']}_{tag}_seed${{seed}}"

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {ENTRY} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suf = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suf}")
    return tag, "\n".join(lines)


def build_script_text(case, sigma_lr, seed, cuda_device):
    tag, cmd = build_command(case, sigma_lr, cuda_device)
    return tag, "\n".join([
        "#!/bin/bash", "", "set -e", "",
        f"sigma_lr={sigma_lr}", f"seed={seed}", "",
        cmd, "",
    ])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    # seed-outermost so each seed completes a full coverage before next starts
    for seed in SEEDS:
        for case in CASES:
            for slr in SIGMA_LR_VALUES:
                jobs.append((case, slr, seed))

    generated = []
    for idx, (case, slr, seed) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        tag, text = build_script_text(case, slr, seed, gpu)
        name = f"{case['case_name']}_{tag}_seed{seed}.sh"
        path = OUTPUT_DIR / name
        path.write_text(text, encoding="utf-8")
        make_executable(path)
        generated.append((path, gpu))
        print(f"Generated: {path}  [GPU {gpu}]")

    total = len(generated)
    print(f"\nGenerated {total} scripts (spectral_aadmm × 9 cells × 3 σ × 3 seeds).")

    if not RUN_AFTER_GENERATION:
        return

    max_workers = len(CUDA_DEVICES) * MAX_PARALLEL_PER_GPU
    print(f"\nLaunching {max_workers} workers...\n")
    gpu_sems = {g: threading.Semaphore(MAX_PARALLEL_PER_GPU) for g in CUDA_DEVICES}
    print_lock = threading.Lock()

    def run_one(script_path, gpu):
        log_path = LOG_DIR / f"{script_path.stem}.log"
        with gpu_sems[gpu]:
            with print_lock:
                print(f"Launching: {script_path.name} [GPU {gpu}] -> {log_path}")
            with open(log_path, "w") as f:
                p = subprocess.Popen(["bash", str(script_path)],
                                     stdout=f, stderr=subprocess.STDOUT)
                ret = p.wait()
        return script_path, log_path, ret

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, sp, gpu) for sp, gpu in generated]
        for fut in as_completed(futs):
            sp, lp, ret = fut.result()
            done += 1
            with print_lock:
                if ret == 0:
                    print(f"[{done}/{total}] Finished: {sp.name}")
                else:
                    print(f"[{done}/{total}] FAILED: {sp.name} (exit {ret})")
                    failed.append((sp, ret, lp))

    if failed:
        print("\nFailed:")
        for sp, code, lp in failed:
            print(f"  {sp} (exit {code}) -> {lp}")
    else:
        print("\nAll scripts completed.")


if __name__ == "__main__":
    main()
