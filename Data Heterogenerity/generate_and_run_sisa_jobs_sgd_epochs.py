"""Generator for exact-ADMM SGD sweeps.

CHANGE LOG
- 2026-04-24: Added cifar10 x {label1, label2, label3} cases. Motivation:
  the Lipschitz-floor path plateaus at ~0.21 on cifar10 simple-cnn while
  tuned SISA reaches ~0.40, so we need to test whether the exact-ADMM
  adaptive-sigma path (which has NO Lipschitz floor, just online_convex_bal)
  works on cifar10.
- 2026-04-24 (later): switched experiment_sisa_practise_admm.py's local solve
  to WARM-START from w_i^{k-1} (the previous local solution) instead of
  resetting to w_global^k each round. The caller already prepared the model
  with W_b_initial[sb]; the inner reset that killed the warm-start was
  removed. See the function docstring of `local_admm_train` in
  experiment_sisa_practise_admm.py for the design note.
  Re-running the full sweep with the new behavior in a fresh wandb project
  (`sisa-exact-admm-warmstart`) to keep results clean and comparable to the
  pre-warm-start runs in `sisa-exact-admm-sgd-epochs-4-22`. mnist/fmnist
  cases are uncommented because the warm-start might shift their numbers
  too -- need the full grid to validate / characterize the change.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_exact_admm_warmstart")
LOG_DIR = OUTPUT_DIR / "logs"
# LOCAL_METRICS_DIR = OUTPUT_DIR / "local_metrics"

# physical GPU ids to distribute work across (all 8 H100s)
CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]

# Base sweep seeds
SEEDS = [0, 1, 2]

# Extra seeds to run ONLY on the headline cells, to get tight CIs on the
# fmnist-label3-beats-SISA claim. Key tuple: (case, method, sigma_lr, epochs, lr).
# 2026-04-24 (later): commented out for the warm-start sweep -- we want a
# clean 3-seed grid first, then add headline extras only after the warm-start
# results are evaluated. Restore by uncommenting below.
EXTRA_SEEDS = [3, 4, 5, 6, 7, 8, 9]
HEADLINE_CELLS = set()
# HEADLINE_CELLS = {
#     ("mnist_label3_n10",  "sgd_adaptive", "1e4", "3", "0.001"),
#     ("fmnist_label3_n10", "sgd_adaptive", "1e4", "3", "0.001"),
#     ("mnist_label3_n10",  "sgd_original", "1e3", "1", "0.001"),
#     ("fmnist_label3_n10", "sgd_original", "1e3", "1", "0.001"),
# }

# Concurrent runs per physical GPU. simple-cnn batch=64 uses ~1-2GB VRAM
# and a tiny fraction of H100 SMs per process. At 16/GPU we put ~32GB on
# each (out of 80GB), CUDA contexts ~6.4GB; comfortable. With 128 workers
# total across 8 GPUs, 352 jobs finish in ~3 waves vs ~5.5 at 8/GPU.
# Drop to 12 or 8 if `nvidia-smi` shows OOM or job startup contention.
MAX_PARALLEL_PER_GPU = 16

# Sweep over initial sigma
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

# Sweep over local epochs
EPOCHS_VALUES = ["1", "3", "10"]

# Sweep over local learning rate
LR_VALUES = ["0.001"]

EXACT_ADMM_ENTRY = "experiment_sisa_practise_admm.py"
ORIGINAL_ENTRY = "experiment_sisa_practise_wandb.py"

# Args for the adaptive (sgd_adaptive) entry point. Uses --optimizer=sgd
# and the adaptive sigma machinery defined in ADAPTIVE_EXTRA_ARGS.
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
    "optimizer": "sgd",
    "use_wandb": "true",
    "wandb_project": "sisa-exact-admm-warmstart",
    # "local_log_dir": str(LOCAL_METRICS_DIR),
}

# Args for the original (sgd_original) entry point. Mirrors run_sisa_cifar.sh
# exactly, only dataset / partition / sigma_lr (and seed) are swept. No
# --optimizer flag (the original sisa branch uses a manual RMSProp-like
# update and ignores args.optimizer). Wandb flags added on top so runs
# still show up on the dashboard.
ORIGINAL_COMMON_ARGS = {
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
    "rho_lr": "1e2",
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "sisa-exact-admm-warmstart",
}

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

# 2026-04-24 (later): full sweep on mnist + fmnist + cifar10 enabled because
# the local-solve warm-start change could shift mnist/fmnist numbers too.
# Going to a fresh wandb_project so old (reset-to-w_global) and new
# (warm-start) results don't get mixed up.
CASES = [
    {"case_name": "mnist_label3_n10",  "dataset": "mnist",   "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "fmnist_label3_n10", "dataset": "fmnist",  "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "mnist_label2_n10",  "dataset": "mnist",   "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "fmnist_label2_n10", "dataset": "fmnist",  "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "mnist_label1_n10",  "dataset": "mnist",   "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "fmnist_label1_n10", "dataset": "fmnist",  "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "cifar10_label1_n10","dataset": "cifar10", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "cifar10_label2_n10","dataset": "cifar10", "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "cifar10_label3_n10","dataset": "cifar10", "partition": "noniid-#label3", "model": "simple-cnn"},
]

# Original SISA-ADMM baseline (no sigma_mode logic). Uses experiment_sisa_practise_wandb.py.
# All args are folded into ORIGINAL_COMMON_ARGS above; no extras needed.
METHODS = [
    {
        "method_name": "sgd_adaptive",
        "entry": EXACT_ADMM_ENTRY,
        "base_args": COMMON_ARGS,
        "extra_args": ADAPTIVE_EXTRA_ARGS,
        "sweep_sigma": True,
        "sweep_epochs": True,
        "sweep_lr": True,
    },
    {
        "method_name": "sgd_original",
        "entry": ORIGINAL_ENTRY,
        "base_args": ORIGINAL_COMMON_ARGS,
        "extra_args": {},
        "sweep_sigma": True,
        "sweep_epochs": False,  # original ignores args.epochs; pinned to 1
        "sweep_lr": False,      # pinned to 0.001 (matches run_sisa_cifar.sh)
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


def make_experiment_tag(sigma_lr_val: str, epochs_val: str, lr_val: str) -> str:
    return f"sgd_ep{epochs_val}_lr{lr_val}_sig{sigma_lr_val}_4_22"


def build_wandb_names(case: dict, method_name: str, tag: str):
    group = f"{case['case_name']}-sisa-{tag}"
    run_name = f"{case['dataset']}_sig${{sigma_lr}}_{method_name}_{tag}_seed${{seed}}"
    return group, run_name


def build_command_template(case: dict, method: dict, tag: str, cuda_device: str = "0",
                           epochs: str = "1", lr: str = "0.001") -> str:
    args = {}
    args.update(method.get("base_args", COMMON_ARGS))
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "alg": "sisa",
        "partition": case["partition"],
        "epochs": epochs,
        "lr": lr,
    })
    args.update(method["extra_args"])

    wandb_group, wandb_run_name = build_wandb_names(case, method["method_name"], tag=tag)
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {method['entry']} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(case: dict, method: dict, sigma_lr: str, tag: str, seed: int,
                      cuda_device: str = "0", epochs: str = "1", lr: str = "0.001") -> str:
    cmd = build_command_template(case, method, tag=tag, cuda_device=cuda_device,
                                 epochs=epochs, lr=lr)
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
    # LOCAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all job specs first, then assign GPUs round-robin
    jobs = []

    for case in CASES:
        for method in METHODS:
            sigma_values = SIGMA_LR_VALUES if method.get("sweep_sigma", False) else ["1e2"]
            epochs_values = EPOCHS_VALUES if method.get("sweep_epochs", False) else ["1"]
            lr_values = LR_VALUES if method.get("sweep_lr", False) else ["0.001"]

            for slr in sigma_values:
                for ep in epochs_values:
                    for lr in lr_values:
                        key = (case["case_name"], method["method_name"], slr, ep, lr)
                        seeds_for_cell = list(SEEDS)
                        if key in HEADLINE_CELLS:
                            seeds_for_cell.extend(EXTRA_SEEDS)
                        for seed in seeds_for_cell:
                            tag = make_experiment_tag(slr, ep, lr)
                            script_name = f"{case['case_name']}_{method['method_name']}_{tag}_seed{seed}.sh"
                            jobs.append((case, method, slr, ep, lr, tag, seed, script_name))

    # Round-robin GPU assignment (one script = one seed)
    generated_scripts = []
    for idx, (case, method, slr, ep, lr, tag, seed, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            case, method, sigma_lr=slr, tag=tag, seed=seed,
            cuda_device=gpu, epochs=ep, lr=lr,
        )
        script_path.write_text(script_text, encoding="utf-8")
        make_executable(script_path)
        generated_scripts.append((script_path, gpu))
        print(f"Generated: {script_path}  [GPU {gpu}]")

    total = len(generated_scripts)
    print(f"\nGenerated {total} single-seed scripts across {len(CUDA_DEVICES)} GPUs.")
    print(f"Headline cells ({len(HEADLINE_CELLS)}) get seeds {SEEDS + EXTRA_SEEDS}; "
          f"others get seeds {SEEDS}.")

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
    total = len(generated_scripts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, sp, gpu) for sp, gpu in generated_scripts]
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
