"""Pilot for AdamW-style decoupled ADMM regularizer.

Tests two new optimizer modes added to experiment_sisa_practise_admm.py
(2026-04-24): `adamw_admm_explicit` and `adamw_admm_implicit`. Both
decouple the ADMM regularizer alpha*(pi + sigma*(w - w_g)) from Adam's
moment estimates m, v -- the same trick AdamW uses for L2 weight decay.
This avoids the sigma-cancellation pathology in plain `adam_warmstart`,
where v absorbs the sigma-penalty gradient and the effective step
becomes ~sign(w - w_g), independent of sigma.

See notes/adam_warmstart_pseudocode.tex Algorithms 3 and 4 for the math.

What the pilot tests
--------------------
Headline question: does decoupling the regularizer (a) preserve
sigma-robustness on mnist/fmnist label1 and (b) close the cifar10
ceiling gap relative to tuned SISA at sigma=1e3?

Cells: 5 -- {mnist, fmnist} x label1 (extreme heterogeneity, σ-robustness
testbed) plus cifar10 x {label1, label2, label3} (the regime where
adam_warmstart and the Lipschitz floor both fail).

sigma: {1e2, 1e3, 1e4}, ep: {1, 3}, seeds: {0, 1, 2}, variants:
{explicit, implicit}. Total: 5 x 3 x 2 x 3 x 2 = 180 runs.

Reset / warm-start: local_init='reset' (the default after the
2026-04-24 revert). admm_reg_lr is set to args.lr by default; explicit
variant may oscillate at sigma=1e4 (admm_reg_lr * sigma = 10), implicit
is unconditionally stable.

Comparison baselines (no need to re-run):
  - adam_warmstart on these cells lives in sisa-exact-admm-sgd-epochs-4-22.
  - sgd reset baseline lives in sisa-exact-admm-sgd-epochs-4-22 (same project).
The new project `sisa-exact-admm-adamw-pilot` keeps these results
isolated for the dashboard's `warmstart_vs_reset`-style comparison.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_adamw_pilot")
LOG_DIR = OUTPUT_DIR / "logs"

# All 8 H100s
CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]

# Concurrent runs per GPU. simple-cnn batch=64 is tiny; 8 fits comfortably.
# Bump to 12-16 if nvidia-smi shows low SM util.
MAX_PARALLEL_PER_GPU = 8

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]
EPOCHS_VALUES = ["1", "3"]
LR_VALUES = ["0.001"]

# Two new optimizer variants.
OPTIMIZER_VARIANTS = ["adamw_admm_explicit", "adamw_admm_implicit"]

EXACT_ADMM_ENTRY = "experiment_sisa_practise_admm.py"

# 5 cells: 2 headline (mnist/fmnist label1) + 3 cifar10 partitions.
CASES = [
    {"case_name": "mnist_label1_n10",   "dataset": "mnist",   "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "fmnist_label1_n10",  "dataset": "fmnist",  "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "cifar10_label1_n10", "dataset": "cifar10", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "cifar10_label2_n10", "dataset": "cifar10", "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "cifar10_label3_n10", "dataset": "cifar10", "partition": "noniid-#label3", "model": "simple-cnn"},
]

# Args common to every run. Matches existing exact-ADMM SGD sweeps so
# results are directly comparable on the dashboard.
COMMON_ARGS = {
    "batch-size": "64",
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
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "sisa-exact-admm-adamw-pilot",
    # Local-solve init: reset to w_global each round (default behavior).
    "local_init": "reset",
    # Decoupled regularizer stepsize. Defaults to args.lr if unset.
    # Explicit variant: stable when admm_reg_lr * sigma is O(1). With
    # admm_reg_lr=1e-3 and sigma=1e4, this product is 10 -- the explicit
    # variant may struggle there. Implicit is stable for any sigma.
    "admm_reg_lr": "0.001",
}

# Adaptive sigma machinery (online convex balanced; no Lipschitz floor in admm.py).
ADAPTIVE_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal",
    "sigma_min": "1e-6",
    "sigma_max": "1e4",
    "eta_u": "0.05",
    "G_clip": "5.0",
    "eps": "1e-12",
    "sigma_update_freq": "1",
    "sigma_ema_beta": "0.9",
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


def make_tag(optimizer: str, sigma_lr: str, ep: str, lr: str) -> str:
    # Shorten optimizer name for filenames: adamw_admm_explicit -> adw_exp.
    short = {
        "adamw_admm_explicit": "adw_exp",
        "adamw_admm_implicit": "adw_imp",
    }[optimizer]
    return f"{short}_ep{ep}_lr{lr}_sig{sigma_lr}"


def build_command(case: dict, optimizer: str, tag: str,
                  cuda_device: str, ep: str, lr: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update(ADAPTIVE_EXTRA_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "alg": "sisa",
        "partition": case["partition"],
        "epochs": ep,
        "lr": lr,
        "optimizer": optimizer,
    })
    args["wandb_group"] = f"{case['case_name']}-{tag}"
    args["wandb_run_name"] = f"{case['dataset']}_{tag}_seed${{seed}}"

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {EXACT_ADMM_ENTRY} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(case: dict, optimizer: str, sigma_lr: str, ep: str, lr: str,
                      seed: int, tag: str, cuda_device: str) -> str:
    cmd = build_command(case, optimizer=optimizer, tag=tag,
                        cuda_device=cuda_device, ep=ep, lr=lr)
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
        for optimizer in OPTIMIZER_VARIANTS:
            for slr in SIGMA_LR_VALUES:
                for ep in EPOCHS_VALUES:
                    for lr in LR_VALUES:
                        for seed in SEEDS:
                            tag = make_tag(optimizer, slr, ep, lr)
                            script_name = f"{case['case_name']}_{tag}_seed{seed}.sh"
                            jobs.append((case, optimizer, slr, ep, lr, seed, tag, script_name))

    generated = []
    for idx, (case, optimizer, slr, ep, lr, seed, tag, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            case, optimizer=optimizer, sigma_lr=slr, ep=ep, lr=lr,
            seed=seed, tag=tag, cuda_device=gpu,
        )
        script_path.write_text(script_text, encoding="utf-8")
        make_executable(script_path)
        generated.append((script_path, gpu, optimizer))
        print(f"Generated: {script_path}  [GPU {gpu}]  ({optimizer})")

    total = len(generated)
    by_opt = {}
    for _, _, opt in generated:
        by_opt[opt] = by_opt.get(opt, 0) + 1
    print(f"\nGenerated {total} single-seed scripts.")
    for opt, n in by_opt.items():
        print(f"  {opt}: {n}")

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
