"""Local-solver comparison: Adam-family half (companion to ..._sgd.py).

Question: when the local solve uses Adam (warmstart vs cold) or AdamW
(decoupled regularizer, warmstart vs cold), does the OGD sigma rule
remain effective, and does the Lipschitz floor close any gap? All runs
at ep=10 to give Adam-family solvers room to converge.

NAMING NOTE (important for plot/pdf generators):
  These runs use OGD on `experiment_sisa_practise_admm.py` -- the residuals
  are computed from the augmented-Lagrangian local solve, NOT from the SISA
  closed-form local solve in `_online.py`. Label the methods as
  `ogd_admm` / `ogd_admm_lipschitz` to distinguish from `ogd_sisa` /
  `ogd_sisa_lipschitz`.

Specs (16 sub-specs, 2x2x2x2 grid):
  axis 1: sigma rule        - {OGD, OGD + Lipschitz}
  axis 2: local solver      - {adam, adamw_admm_explicit}
  axis 3: optim-state init  - {warmstart, cold}
  axis 4: local epochs      - {3, 10}

Cells: mnist/fmnist/cifar10 x label1/2/3
sigma_0: {1e2, 1e3, 1e4}
seeds: {0, 1, 2}

Total: 16 sub-specs x 9 cells x 3 sigma x 3 seeds = 1296 runs.

Loop order: seed -> spec -> case -> sigma. Putting seed outermost means
each seed completes a full pass over all (spec, case, sigma) before the
next seed starts -- so a partial sweep still has one full coverage of
every config per seed completed.

AdamW reg-step (specs 5-8): hardcoded eta_r = lr / max(sigma, 1), per-batch
post-Adam. Per-batch shrinkage rate = lr * alpha (sigma-invariant); fixes the
prior collapse pathology. No tunable knobs. See _admm.py CHANGE LOG
(2026-05-02 RE-INTRODUCED).

Wandb project: exact-admm-local-solver-adam
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_local_solver_adam")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3"]
MAX_PARALLEL_PER_GPU = 5  # ep=10 with adam family is heavier than sgd

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]
EPOCH_VALUES = ["3", "10"]

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
    "wandb_project": "exact-admm-local-solver-adam",
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "eta_u": "0.05",
    "eta_u_decay": "textbook_sc",
    "G_clip": "5.0",
    "sigma_update_freq": "1",
}

OGD_ARGS = {"sigma_mode": "online_convex_bal"}

OGD_LIPSCHITZ_ARGS = {
    "sigma_mode": "online_convex_bal_lipschitz",
    "lipschitz_estimator": "ema",
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
    "lipschitz_floor_alpha": "1.0",
}

# (optimizer_value, short_label) pairs.
SOLVERS = [
    ("adam_warmstart",                  "adam_ws"),
    ("adam",                            "adam_cold"),
    ("adamw_admm_explicit_warmstart",   "adamw_ws"),
    ("adamw_admm_explicit",             "adamw_cold"),
]

JOB_SPECS = []
for ep in EPOCH_VALUES:
    for opt_val, opt_label in SOLVERS:
        JOB_SPECS.append({
            "spec_id": f"ogd_admm_{opt_label}_ep{ep}",
            "extra_args": {**OGD_ARGS, "optimizer": opt_val, "epochs": ep},
            "tag": lambda slr, _l=opt_label, _ep=ep: f"ogd_admm_{_l}_ep{_ep}_sig{slr}",
        })
        JOB_SPECS.append({
            "spec_id": f"ogd_admm_lipschitz_{opt_label}_ep{ep}",
            "extra_args": {**OGD_LIPSCHITZ_ARGS, "optimizer": opt_val, "epochs": ep},
            "tag": lambda slr, _l=opt_label, _ep=ep: f"ogd_admm_lipschitz_{_l}_ep{_ep}_sig{slr}",
        })

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
    # seed is OUTERMOST so each seed completes a full pass over (spec, case,
    # sigma) before the next seed starts. A partial sweep still gives one
    # complete coverage of every config per finished seed.
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
    print(f"\nGenerated {total} scripts.")
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
