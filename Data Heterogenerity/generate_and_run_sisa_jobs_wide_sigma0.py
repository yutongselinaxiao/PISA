"""Wide σ_0 sweep — to break improved_heuristic and demonstrate OGD's
logarithmic descent budget.

Why this exists: on σ_0 ∈ {1e2, 1e3, 1e4} (the canonical-fl grid),
ogd_sisa+lipschitz and improved_heuristic are within 0-5pp on most cells.
At extreme σ_0, the heuristic should fail because:
  - Multiplicative descent (×τ=2 per fire), bounded by k_max=50 fires.
  - Max log-descent budget ≈ k_max * log(τ) ≈ 50 * 0.69 = 35 nats ≈ 15 log-decades.
  - Within first 50 rounds; rarely reaches the productive σ if σ_0 is far.

OGD with textbook_sc step has logarithmic descent budget:
  - eta_t = 1/(2t) with G_clip=10, total log-descent budget = G_clip * log(K)/2
    ≈ 10 * log(500)/2 ≈ 31 log-units. Significantly more headroom.

Sweep: σ_0 ∈ {1e1, 1e5, 1e6} (NEW values not in canonical-fl).
Methods: 5 (original / heuristic / improved_heuristic / ogd_sisa / ogd_sisa+lip).
Cells: same 9 (mnist/fmnist/cifar10 × label1/2/3).
Seeds: 3.

Total: 5 methods × 3 σ × 9 cells × 3 seeds = 405 runs.

Wandb project: paper-canonical-fl-wide-sigma
  - Distinct from `paper-canonical-fl` to keep the extreme-σ_0 sweep
    cleanly separable. Same code paths, just extended σ_0 grid.

Hypothesis the data should test:
  - At σ_0=1e6, heuristic should plateau at val_acc << ogd_sisa+lip val_acc
    (gap should be 30-50pp on cells where both work at σ_0=1e3).
  - This is the headline plot for "OGD beats heuristic at extreme σ_0".
"""
import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_wide_sigma0")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3"]
MAX_PARALLEL_PER_GPU = 8

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e1", "1e5", "1e6"]  # extreme σ_0 NOT in canonical-fl

ONLINE_ENTRY = "experiment_sisa_practise_online.py"
ORIGINAL_ENTRY = "experiment_sisa_practise_wandb.py"
ADMM_ENTRY = "experiment_sisa_practise_admm.py"

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
    "wandb_project": "paper-canonical-fl-wide-sigma",
}

OGD_BASE = {
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "eta_u": "0.05",
    "eta_u_decay": "textbook_sc",
    "G_clip": "5.0",
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
    {
        "spec_id": "original",
        "entry": ORIGINAL_ENTRY,
        "extra_args": {},
        "tag": lambda sigma_lr: f"original_sig{sigma_lr}",
    },
    {
        "spec_id": "heuristic_admm",
        "entry": ADMM_ENTRY,
        "extra_args": {
            "sigma_mode": "heuristic",
            "sigma_min": "1e-6",
            "sigma_max": "1e8",
            "sigma_mu": "10.0",
            "sigma_tau": "2.0",
            "sigma_ema_beta": "0.9",
            "sigma_kmax": "50",
            "sigma_update_freq": "1",
        },
        "tag": lambda sigma_lr: f"heuristic_admm_mu10_tau2_kmax50_sig{sigma_lr}",
    },
    {
        "spec_id": "convex_bal",
        "entry": ONLINE_ENTRY,
        "extra_args": {**OGD_BASE, "sigma_mode": "online_convex_bal"},
        "tag": lambda sigma_lr: f"convex_bal_sig{sigma_lr}",
    },
    {
        "spec_id": "convex_bal_lipschitz",
        "entry": ONLINE_ENTRY,
        "extra_args": {**OGD_BASE, **LIPSCHITZ_FLOOR,
                       "sigma_mode": "online_convex_bal_lipschitz"},
        "tag": lambda sigma_lr: f"convex_bal_lipschitz_sig{sigma_lr}",
    },
]

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


def build_command(spec, case, tag, cuda_device):
    args = dict(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "partition": case["partition"],
    })
    args.update(spec["extra_args"])
    args["wandb_group"] = f"{case['case_name']}-{tag}"
    args["wandb_run_name"] = f"{case['dataset']}_{tag}_seed${{seed}}"

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {spec['entry']} \\"]
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
    # seed-outermost so each seed gives one full pass over (spec, case, σ)
    for seed in SEEDS:
        for spec in JOB_SPECS:
            for case in CASES:
                for slr in SIGMA_LR_VALUES:
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
    by_spec = {}
    for _, _, sid in generated:
        by_spec[sid] = by_spec.get(sid, 0) + 1
    print(f"\nGenerated {total} scripts (4 specs × 9 cells × 3 σ × 3 seeds).")
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
