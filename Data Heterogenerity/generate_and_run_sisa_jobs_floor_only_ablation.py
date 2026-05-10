"""Floor-without-OGD ablation — set σ_t = α·L̂_t directly each round.

This ablation is the reviewer-protective answer to: "is the OGD step on
log(σ) doing useful work, or could you just track the BB-Lipschitz
estimate?". It uses the SAME BB-Lipschitz state machinery as
`online_convex_bal_lipschitz` (gradient pre-pass at z^k, EMA on BB ratios),
but skips the OGD step entirely:

  σ_t = α · L̂_t                               (NEW: floor_only)
vs.
  u_raw  = u − η · 2·(u − target)              (OGD step)
  u_new  = max(u_raw, log(α·L̂_t))             (with floor projection)
  σ_t    = exp(u_new)                          (online_convex_bal_lipschitz)

Hypothesis the data should test:
  - If floor_only ≈ ogd_lipschitz: OGD adds nothing on top of the floor;
    contribution simplifies to "track L̂".
  - If floor_only < ogd_lipschitz by 3-10pp: OGD adds real value above the
    floor; both components matter.

Sweep:
  - 9 cells (mnist/fmnist/cifar10 × label1/2/3)
  - σ_0 ∈ {1e2, 1e3, 1e4} (matches canonical-fl grid for direct comparison)
  - 3 seeds
  - Single method (`floor_only`)
  - SGD ep=1 (the canonical-fl default)

Total: 1 method × 9 cells × 3 σ × 3 seeds = 81 runs.

Wandb project: paper-canonical-fl-floor-only-ablation
  - Distinct from `paper-canonical-fl` so the ablation runs are cleanly
    separable from the headline comparison runs.

Implementation note: the `floor_only` mode and the gate extensions for
the BB-grad pre-pass were added to `experiment_sisa_practise_online.py` on
2026-05-08. See `--sigma_mode floor_only` (added to argparse choices) and
the new `elif sigma_mode == "floor_only":` branch.
"""
import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_floor_only_ablation")
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
    "wandb_project": "paper-canonical-fl-floor-only-ablation",
    # Floor-only ablation mode
    "sigma_mode": "floor_only",
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    # Lipschitz floor knobs (consumed by floor_only same as
    # online_convex_bal_lipschitz)
    "lipschitz_estimator": "ema",
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
    "lipschitz_floor_alpha": "1.0",
    "sigma_update_freq": "1",
    # OGD knobs not consumed by floor_only, but argparse expects them
    "eta_u": "0.05",
    "eta_u_decay": "textbook_sc",
    "G_clip": "5.0",
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
    tag = f"floor_only_sig{sigma_lr}"
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
    print(f"\nGenerated {total} scripts (floor_only × 9 cells × 3 σ × 3 seeds).")

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
