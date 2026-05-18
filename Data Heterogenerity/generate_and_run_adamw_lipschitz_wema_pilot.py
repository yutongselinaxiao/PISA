"""Polyak local-iterate EMA pilot for AdamW-explicit + Lipschitz floor.

Tests whether smoothing the local iterate w (EMA across batches within a
local round) closes the label1 gap for AdamW-explicit+lip. Hypothesis:
each batch on label1 sees a single class, so the raw post-training w is
biased toward that class; averaging across batches before global
aggregation reduces this bias.

Method change: `--local_weight_ema_beta=0.99` is added to the existing
adamw_admm_explicit{_warmstart} + Lipschitz floor command lines. β=0
(default) preserves the prior behavior exactly; β=0.99 maintains a
Polyak-style EMA on w throughout local training and returns the EMA to
the global aggregation.

Scope: the same 2 specs × 9 cells × 3 σ × 3 seeds = 162 runs the box plot
needs for the AdamW+lip comparison. Wandb project is SEPARATE
(`paper-adamw-explicit-lip-wema-pilot`) so the EMA results are cleanly
comparable to the existing `exact-admm-local-solver-adam` runs without
the EMA.

Resumable: queries wandb for finished/running runs in the pilot project
and only emits scripts for the missing cells, so the launcher can be
re-run after interruptions.

ep=3 only (matches box-plot canonical comparison).
"""
import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb

ENTITY = "selinayutongxiao-university-of-southern-californ"
PROJECT = "paper-adamw-explicit-lip-wema-pilot"

OUTPUT_DIR = Path("generated_adamw_lipschitz_wema")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3"]
MAX_PARALLEL_PER_GPU = 3  # adam ep=3 is moderately heavy

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]
EPOCH_VALUES = ["3"]

LOCAL_WEIGHT_EMA_BETA = "0.99"  # the knob this pilot is testing

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
    "wandb_project": PROJECT,
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "eta_u": "0.05",
    "eta_u_decay": "textbook_sc",
    "G_clip": "5.0",
    "sigma_update_freq": "1",
    "local_weight_ema_beta": LOCAL_WEIGHT_EMA_BETA,
}

OGD_LIPSCHITZ_ARGS = {
    "sigma_mode": "online_convex_bal_lipschitz",
    "lipschitz_estimator": "ema",
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
    "lipschitz_floor_alpha": "1.0",
}

# Only the +Lipschitz variants are in scope for this pilot (the EMA fix
# targets the failure mode of AdamW-explicit+lip on label1). The no-floor
# variants don't have the failure to fix.
SOLVERS = [
    ("adamw_admm_explicit_warmstart",   "adamw_ws_wema"),
    ("adamw_admm_explicit",             "adamw_cold_wema"),
]

JOB_SPECS = []
for ep in EPOCH_VALUES:
    for opt_val, opt_label in SOLVERS:
        JOB_SPECS.append({
            "spec_id": f"ogd_admm_lipschitz_{opt_label}_ep{ep}",
            "extra_args": {**OGD_LIPSCHITZ_ARGS, "optimizer": opt_val, "epochs": ep},
            "tag": lambda slr, _l=opt_label, _ep=ep: f"ogd_admm_lipschitz_{_l}_ep{_ep}_sig{slr}",
        })

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


def expected_run_name(case, tag, seed):
    """Mirrors the wandb_run_name format `{dataset}_{tag}_seed{seed}`."""
    return f"{case['dataset']}_{tag}_seed{seed}"


def fetch_existing_runs(skip_running=True):
    """Returns set of run names already finished (or currently running) in
    the pilot project, so the resumable launcher skips them.

    If the project doesn't exist yet (first launch) returns an empty set.
    """
    api = wandb.Api(timeout=60)
    try:
        runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=500))
    except Exception as e:
        print(f"  (project not found / no runs yet: {e})")
        return set()
    skip = set()
    state_counts = {}
    for r in runs:
        state_counts[r.state] = state_counts.get(r.state, 0) + 1
        if r.state == "finished":
            skip.add(r.name)
        elif skip_running and r.state == "running":
            skip.add(r.name)
    print(f"  wandb states: {state_counts}")
    print(f"  skipping {len(skip)} runs (finished{', + running' if skip_running else ''})")
    return skip


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

    print(f"Querying wandb project '{PROJECT}' for already-finished/running runs ...")
    skip_set = fetch_existing_runs(skip_running=True)

    all_jobs = []
    for seed in SEEDS:
        for spec in JOB_SPECS:
            for case in CASES:
                for slr in SIGMA_LR_VALUES:
                    tag = spec["tag"](slr)
                    name = f"{case['case_name']}_{tag}_seed{seed}.sh"
                    run_name = expected_run_name(case, tag, seed)
                    all_jobs.append((spec, case, slr, seed, tag, name, run_name))

    missing = [j for j in all_jobs if j[-1] not in skip_set]
    print(f"\nFull grid: {len(all_jobs)}; missing/incomplete: {len(missing)}")

    if not missing:
        print("Nothing to do. Pilot is complete.")
        return

    spec_counts = {}
    for j in missing:
        spec_counts[j[0]["spec_id"]] = spec_counts.get(j[0]["spec_id"], 0) + 1
    print("\nMissing runs by spec:")
    for sid, n in sorted(spec_counts.items()):
        print(f"  {sid}: {n}")

    generated = []
    for idx, (spec, case, slr, seed, tag, name, run_name) in enumerate(missing):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        path = OUTPUT_DIR / name
        path.write_text(build_script_text(spec, case, slr, seed, tag, gpu),
                        encoding="utf-8")
        make_executable(path)
        generated.append((path, gpu, spec["spec_id"]))

    total = len(generated)
    print(f"\nGenerated {total} missing scripts.")

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
        print("\nAll missing scripts completed.")


if __name__ == "__main__":
    main()
