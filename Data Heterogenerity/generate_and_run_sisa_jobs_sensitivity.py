"""One-factor-at-a-time (OFAT) sensitivity sweep for online_convex_bal_lipschitz.

Goal: demonstrate that the four "real" tunables (eta_u, G_clip, lipschitz
window, lipschitz EMA beta) all give a tight test-acc band when swept over
a wide range — so the algorithm is easy to tune in practice. The other
knobs (sigma bounds, lipschitz_min_dz/max, eta_u_decay=textbook_sc,
floor_alpha=1.0, estimator=ema) are inert numerical guards or theory-
defaulted, so they are NOT swept here.

Pinned context (one cell, one sigma_0, 3 seeds):
  - cell: fmnist_label2_n10  (middle difficulty)
  - sigma_lr: 1e3            (middle of {1e2, 1e3, 1e4})
  - seeds: {0, 1, 2}

Defaults for the four real tunables (matching canonical_fl OGD config):
  eta_u = 0.05,  G_clip = 5.0,
  lipschitz_window_size = 20,  lipschitz_ema_beta = 0.9

Sweep design (OFAT — one knob varies, others pinned at default):
  eta_u                 : {0.01, 0.05*, 0.1, 0.5}
  G_clip                : {1.0,  5.0*, 20.0, 1e6}     # 1e6 ~= no clip
  lipschitz_window_size : {5,    10,   20*,  50}
  lipschitz_ema_beta    : {0.5,  0.7,  0.9*, 0.99}

The default configuration (* in every dim) is run ONCE as the shared anchor;
each non-default value is one extra config. Total:
  1 default + 4 dims x 3 non-default = 13 configs x 3 seeds = 39 runs.

Wandb project: paper-canonical-fl-sensitivity
  (separate from paper-canonical-fl so the 39 single-cell runs don't dilute
   the headline 4-method comparison dashboard).

Plotting protocol per dim: x = knob value, y = final test acc averaged over
3 seeds, with horizontal default line. Tight band -> easy to tune.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_sensitivity")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 8

SEEDS = [0, 1, 2]
SIGMA_LR = "1e3"

ENTRY = "experiment_sisa_practise_online.py"

CASE = {"case_name": "fmnist_label2_n10",
        "dataset": "fmnist",
        "partition": "noniid-#label2",
        "model": "simple-cnn"}

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
    "sigma_lr": SIGMA_LR,
    "rho_lr": "1e2",
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "paper-canonical-fl-sensitivity",
    "model": CASE["model"],
    "dataset": CASE["dataset"],
    "partition": CASE["partition"],
    "sigma_mode": "online_convex_bal_lipschitz",
    # inert / theory-defaulted knobs (not swept)
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "eta_u_decay": "textbook_sc",
    "lipschitz_estimator": "ema",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
    "lipschitz_floor_alpha": "1.0",
}

DEFAULTS = {
    "eta_u": "0.05",
    "G_clip": "5.0",
    "lipschitz_window_size": "20",
    "lipschitz_ema_beta": "0.9",
}

# Each dim: list of non-default values (the default is added separately as the
# shared anchor config so it's only run once).
SWEEP_DIMS = {
    "eta_u":                 ["0.01", "0.1", "0.5"],
    "G_clip":                ["1.0",  "20.0", "1e6"],
    "lipschitz_window_size": ["5",    "10",   "50"],
    "lipschitz_ema_beta":    ["0.5",  "0.7",  "0.99"],
}


def slug(value: str) -> str:
    """Filename-safe slug for a numeric value (0.05 -> 0p05, 1e6 -> 1e6)."""
    return value.replace(".", "p").replace("+", "")


def build_configs():
    """Return list of (config_id, overrides_dict).

    config_id encodes which dim is varied and to what value. The shared
    default appears once with id 'default'.
    """
    configs = [("default", dict(DEFAULTS))]
    for dim, values in SWEEP_DIMS.items():
        for v in values:
            cfg = dict(DEFAULTS)
            cfg[dim] = v
            cid = f"{dim}_{slug(v)}"
            configs.append((cid, cfg))
    return configs


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


def build_command(config_id, overrides, cuda_device):
    args = dict(COMMON_ARGS)
    args.update(overrides)
    tag = f"sens_{config_id}_sig{SIGMA_LR}"
    args["wandb_group"] = f"{CASE['case_name']}-{tag}"
    args["wandb_run_name"] = f"{CASE['dataset']}_{tag}_seed${{seed}}"

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {ENTRY} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suf = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suf}")
    return "\n".join(lines), tag


def build_script_text(config_id, overrides, seed, cuda_device):
    cmd, _ = build_command(config_id, overrides, cuda_device)
    return "\n".join([
        "#!/bin/bash", "", "set -e", "",
        f"seed={seed}", "",
        cmd, "",
    ])


RUN_AFTER_GENERATION = True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    jobs = []
    for cid, overrides in configs:
        for seed in SEEDS:
            tag = f"sens_{cid}_sig{SIGMA_LR}"
            name = f"{CASE['case_name']}_{tag}_seed{seed}.sh"
            jobs.append((cid, overrides, seed, name))

    generated = []
    for idx, (cid, overrides, seed, name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        path = OUTPUT_DIR / name
        path.write_text(build_script_text(cid, overrides, seed, gpu),
                        encoding="utf-8")
        make_executable(path)
        generated.append((path, gpu, cid))
        print(f"Generated: {path}  [GPU {gpu}]  ({cid})")

    total = len(generated)
    print(f"\nGenerated {total} scripts ({len(configs)} configs x {len(SEEDS)} seeds).")
    by_cfg = {}
    for _, _, cid in generated:
        by_cfg[cid] = by_cfg.get(cid, 0) + 1
    for cid, n in by_cfg.items():
        print(f"  {cid}: {n}")

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
