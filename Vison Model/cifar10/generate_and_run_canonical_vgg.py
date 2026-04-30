"""Canonical VGG-11 / CIFAR-10 σ-rule comparison.

Four methods × σ_0 ∈ {4.5, 1e2, 1e3, 1e4} × 3 seeds = 48 runs.
σ_0 = 4.5 is the paper-tuned VGG anchor (readme.md).

  1. original              → fixed lr-coupled schedule (paper baseline at σ_0=4.5)
  2. heuristic             → Boyd μ=10, τ=2 multiplicative
  3. convex_bal            → OGD on log(σ), no floor
  4. convex_bal_lipschitz  → OGD + BB Lipschitz floor

Uses l2_lower_loss_mPiAM_varying_rho_sigma_lipschitz.py with the canonical
RMS residual aggregation (fixed 2026-04-30) and paper-faithful VGG hyperparams
(radam optimizer, rho_lr=3e4, decay_epoch=10, lr-gamma=0.9, weight_decay=2.5e-4,
beta_rmsprop=0.995, l2_lambda=4e-4 — all from readme.md).

Wandb project: paper-canonical-vision-vgg
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_canonical_vgg")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 2  # VGG-11 lighter than ResNet-34

SEEDS = [0, 1, 2]
# 4.5 is paper-tuned VGG anchor; 1e2/1e3/1e4 probe σ_0-robustness.
SIGMA_LR_VALUES = ["4.5", "1e2", "1e3", "1e4"]
SIGMA_UPDATE_FREQ = "391"

ENTRY = "l2_lower_loss_mPiAM_varying_rho_sigma_lipschitz.py"

CASES = [
    {"case_name": "cifar10_vgg11", "model": "vgg"},
]

# Mirror readme.md VGG command.
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
    "device": "cuda:0",
    "datadir": "/dataMeR2/yutong/datasets",
    "sigma_lr": "${sigma_lr}",
    "seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "paper-canonical-vision-vgg",
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "G_clip": "5.0",
    "sigma_update_freq": SIGMA_UPDATE_FREQ,
}

OGD_BASE = {
    "eta_u": "0.05",
    "eta_u_decay": "textbook_sc",
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
    {"spec_id": "original",
     "extra_args": {"sigma_mode": "fixed"},
     "tag": lambda slr: f"original_sig{slr}"},
    {"spec_id": "heuristic",
     "extra_args": {"sigma_mode": "heuristic", "heuristic_mu": "10.0", "heuristic_tau": "2.0"},
     "tag": lambda slr: f"heuristic_mu10_tau2_sig{slr}"},
    {"spec_id": "convex_bal",
     "extra_args": {**OGD_BASE, "sigma_mode": "online_convex_bal"},
     "tag": lambda slr: f"convex_bal_sig{slr}"},
    {"spec_id": "convex_bal_lipschitz",
     "extra_args": {**OGD_BASE, **LIPSCHITZ_FLOOR,
                    "sigma_mode": "online_convex_bal_lipschitz"},
     "tag": lambda slr: f"convex_bal_lipschitz_sig{slr}"},
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


def build_wandb_names(case, tag):
    return f"{case['case_name']}-{tag}", f"{case['case_name']}_{tag}_seed${{seed}}"


def build_command(spec, case, tag, cuda_device):
    args = {}
    args.update(COMMON_ARGS)
    args.update({"model": case["model"]})
    args.update(spec["extra_args"])
    grp, name = build_wandb_names(case, tag)
    args["wandb_group"] = grp
    args["wandb_run_name"] = name
    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {ENTRY} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suf = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suf}")
    return "\n".join(lines)


def build_script_text(spec, case, slr, seed, tag, cuda_device):
    cmd = build_command(spec, case, tag, cuda_device)
    return "\n".join([
        "#!/bin/bash", "", "set -e", "",
        f"sigma_lr={slr}", f"seed={seed}", "",
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
                    name = f"{case['case_name']}_{tag}_seed{seed}.sh"
                    jobs.append((spec, case, slr, seed, tag, name))

    generated = []
    for idx, (spec, case, slr, seed, tag, name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        path = OUTPUT_DIR / name
        path.write_text(build_script_text(spec, case, slr, seed, tag, gpu),
                        encoding="utf-8")
        make_executable(path)
        generated.append((path, gpu, spec["spec_id"]))
        print(f"Generated: {path}  [GPU {gpu}]  ({spec['spec_id']})")

    total = len(generated)
    print(f"\nGenerated {total} scripts.")
    by_spec = {}
    for _, _, sid in generated:
        by_spec[sid] = by_spec.get(sid, 0) + 1
    for sid, n in by_spec.items():
        print(f"  {sid}: {n}")

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
        futs = [ex.submit(run_one, sp, gpu) for sp, gpu, _ in generated]
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
