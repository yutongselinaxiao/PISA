"""adam_warmstart inner solver paired with OGD + Lipschitz floor sigma rule.

Tests whether the OGD-with-Lipschitz adaptive-sigma machinery (ported from
_online.py to _admm.py) restores sigma-sensitivity when the local solve is
adam_warmstart. Recall: adam_warmstart's m, v, t persist across rounds, and
when Adam sees the full augmented-Lagrangian gradient the sigma-scaled term
contributes equally to m and sqrt(v), so the m/sqrt(v) step becomes
sigma-invariant ("sigma cancellation"). The headline question this sweep
answers:

  Does adapting sigma online (OGD on log(sigma) with the BB Lipschitz floor)
  put the sigma dial back in play even when the local solver is
  adam_warmstart? If sigma-trajectories diverge across sigma_0 but final
  test acc is similar -> floor + OGD is doing the work; if everything still
  collapses to one trajectory -> Adam's normalization eats the dial.

The OGD + Lipschitz config mirrors canonical_fl's `convex_bal_lipschitz`
spec: textbook_sc decay (parameter-free 1/(2k)), EMA Lipschitz estimator
(beta=0.9, window=20), hard projection (alpha=1.0). The local solver is
adam_warmstart instead of the closed-form SISA step (-> uses _admm.py
entry point, not _online.py).

Cells: mnist/fmnist/cifar10 x label1/2/3 x sigma_0 in {1e2, 1e3, 1e4} x
3 seeds = 81 runs.

Wandb project: paper-adamwarmstart-ogd-lipschitz
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_adam_warmstart_ogd_lipschitz")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 4  # adam_warmstart + per-client BB grad pre-pass is heavier

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

ENTRY = "experiment_sisa_practise_admm.py"

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
    "wandb_project": "paper-adamwarmstart-ogd-lipschitz",
    # adam_warmstart inner solver -- the headline pairing.
    "optimizer": "adam_warmstart",
    # OGD + Lipschitz floor sigma rule.
    "sigma_mode": "online_convex_bal_lipschitz",
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "eta_u": "0.05",
    "eta_u_decay": "textbook_sc",
    "G_clip": "5.0",
    "lipschitz_estimator": "ema",
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
    "lipschitz_floor_alpha": "1.0",
    "sigma_update_freq": "1",
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


def build_wandb_names(case: dict, sigma_lr: str):
    tag = f"adam_warmstart_ogd_lipschitz_sig{sigma_lr}"
    group = f"{case['case_name']}-{tag}"
    run_name = f"{case['dataset']}_{tag}_seed${{seed}}"
    return tag, group, run_name


def build_command(case: dict, sigma_lr: str, cuda_device: str):
    args = dict(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "partition": case["partition"],
    })
    tag, grp, name = build_wandb_names(case, sigma_lr)
    args["wandb_group"] = grp
    args["wandb_run_name"] = name

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
    for case in CASES:
        for slr in SIGMA_LR_VALUES:
            for seed in SEEDS:
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
    print(f"\nGenerated {total} scripts ({len(CASES)} cells x "
          f"{len(SIGMA_LR_VALUES)} sigma x {len(SEEDS)} seeds).")

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
