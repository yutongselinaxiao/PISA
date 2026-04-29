"""Vision-model CIFAR-10 + VGG-11 sweep for the Lipschitz-floored online σ
adaptation method.

Sister generator to generate_and_run_vision_cifar10_lipschitz.py (which targets
ResNet-34). Hyperparameters mirror the VGG entry in
PISA/Vison Model/cifar10/readme.md (the paper's recommended VGG-11 command for
Table 3, which produced the 91.25% headline number):

    python l2_lower_loss_mPiAM_varying_rho_sigma.py --model vgg --optim radam \
        --eps 1e-8 --sigma_lr 4.5 --rho_lr 3e4 --beta1 0.9 --beta2 0.999 \
        --momentum 0.9 --batchsize 128 --total_epoch 205 --decay_epoch 10 \
        --lr-gamma 0.9 --baseline_acc 0.9103 --beta_rmsprop 0.995 \
        --weight_decay 2.5e-4 --l2_lambda 4e-4

Uses `l2_lower_loss_mPiAM_varying_rho_sigma_lipschitz.py` as the entry --
the paper-faithful VGG-style training procedure (explicit --l2_lambda L2 on
global w + per-sub-batch gradient-norm normalization in the local solver),
with the σ-OGD + Lipschitz floor machinery wired in. Fixed-mode runs in this
sweep should reproduce the paper's VGG-11 setup; the OGD-mode rows replace
only the σ schedule.

Four methods per cell, all writing to `paper-lipschitz-vision-cifar10-vgg`:
  - Original PISA fixed-σ schedule         (--sigma_mode fixed)
  - Boyd residual-balance heuristic         (--sigma_mode heuristic)
  - OGD on σ, no Lipschitz floor           (--sigma_mode online_convex_bal)
  - OGD on σ + BB Lipschitz floor          (--sigma_mode online_convex_bal_lipschitz)

σ-robustness sweep: σ_0 ∈ {4.5, 1e2, 1e3, 1e4} × 3 seeds × 4 methods = 48 runs.
σ_0 = 4.5 is paper's VGG-tuned anchor (sanity check); 1e2-1e4 probe how each
method degrades when σ_0 is far from the tuned value.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_vision_cifar10_vgg_lipschitz")
LOG_DIR = OUTPUT_DIR / "logs"

# VGG-11 + bs=128 is much lighter than ResNet-34. Bump if you have headroom.
CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 2

SEEDS = [0, 1, 2]
# 4.5 is paper's VGG-tuned anchor; 1e2/1e3/1e4 probe σ₀-robustness.
SIGMA_LR_VALUES = ["4.5", "1e2", "1e3", "1e4"]

ESTIMATOR = "ema"
LIPSCHITZ_WINDOW_SIZE = "20"
ETA_U_DECAY = "textbook_sc"

# CIFAR-10 trainset / batchsize = 50000 / 128 = ~391 batches per epoch.
# Fire one σ-OGD step per epoch.
SIGMA_UPDATE_FREQ = "391"

ENTRY = "l2_lower_loss_mPiAM_varying_rho_sigma_lipschitz.py"

CASES = [
    {"case_name": "cifar10_vgg11", "model": "vgg"},
]

# Mirror the VGG command in readme.md (minus --l2_lambda; see header caveat).
COMMON_ARGS = {
    "optim": "radam",
    "eps": "1e-8",
    "rho_lr": "3e4",
    "beta1": "0.9",
    "beta2": "0.999",
    "momentum": "0.9",
    "batchsize": "128",
    "total_epoch": "205",
    "decay_epoch": "10",
    "lr-gamma": "0.9",
    "weight_decay": "2.5e-4",
    "baseline_acc": "91.03",
    "beta_rmsprop": "0.995",
    "l2_lambda": "4e-4",
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
    "datadir": "/dataMeR2/yutong/datasets",
    # plumbed via shell template substitution
    "sigma_lr": "${sigma_lr}",
    "seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "paper-lipschitz-vision-cifar10-vgg",
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

HEURISTIC_EXTRA_ARGS = {
    "sigma_mode": "heuristic",
    "heuristic_mu": "10.0",
    "heuristic_tau": "2.0",
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
        "spec_id": "heuristic",
        "extra_args": HEURISTIC_EXTRA_ARGS,
        "tag": lambda sigma_lr: f"heuristic_mu10_tau2_sig{sigma_lr}",
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
