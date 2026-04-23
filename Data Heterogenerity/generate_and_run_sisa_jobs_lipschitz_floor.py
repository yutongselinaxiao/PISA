"""Generator for the Lipschitz-floor (hard-projection) adaptive-sigma pilot.

Scripts are distributed round-robin across CUDA_DEVICES and launched in
parallel, gated by a per-GPU semaphore (MAX_PARALLEL_PER_GPU).

The core claim under test: with --sigma_mode online_convex_bal_lipschitz
(hard projection onto {u >= log L_hat}) and eta_u_decay=textbook_sc, the
final test accuracy is insensitive to sigma_0 -- all three initial values
should converge to similar accuracy.

Pilot history:
- 2026-04-19: running_min estimator tested against EMA; EMA won on
  sigma0-robustness and accuracy. Estimator axis dropped (EMA only).
  See memory project_lipschitz_estimator_ema_chosen.md.
- 2026-04-19: decay-schedule pilot. Constant eta_u caused sigma limit cycles
  on fmnist. eta_u_decay=inverse failed at sigma_0=1e4 (too little descent
  budget). Moved to textbook_sc = 1/(mu*k) with mu=2, parameter-free.

CHANGE LOG
- 2026-04-23: wandb audit (see wandb_sweep_findings.md + dashboard section 8)
  showed the only statistically significant parameter-free win is on
  mnist_label1 (+5pp, ~2.5x seed-std). cifar10 Pareto flags are trivial
  because no SGD convex-bal competitor runs exist on cifar10, and the
  textbook_sc cells currently have only 3 UNIQUE seeds (0, 1, 2) each --
  the "6 runs" counted in wandb were seeds 0-2 duplicated twice.
  To tighten CIs on the label1 claim and fill the cifar10 baseline gap,
  this generator now launches TWO disjoint sweeps into a single new
  wandb_project:

  JOB_SPEC "original_cifar":
    * method: sgd_original (experiment_sisa_practise_wandb.py)
    * cases: cifar10 x {label1, label2, label3}
    * seeds: [0, 1, 2]
    * sigma0:  {1e2, 1e3, 1e4}
    * params: derived from run_sisa_cifar.sh (rho_lr=1e2,
      datadir=/data/yutong/datasets); only seed and sigma0 vary.
    * jobs: 3 partitions x 3 sigma x 3 seeds = 27 runs.

  JOB_SPEC "lipschitz_label1_extra":
    * method: online_convex_bal_lipschitz + eta_u_decay=textbook_sc
      (experiment_sisa_practise_online.py)
    * cases: {mnist, fmnist, cifar10} x label1
    * seeds: [3, 4, 5, 6, 7, 8, 9]  -- 7 extra on top of existing 0-2
      to reach n=10 unique seeds per cell.
    * sigma0: {1e2, 1e3, 1e4}
    * jobs: 3 datasets x 3 sigma x 7 seeds = 63 runs.

  Grand total: 90 runs. Launched with 1 seed per script, 8 GPUs x 8 concurrent
  = 64 workers (pattern from generate_and_run_sisa_jobs_sgd_epochs.py).

  Other changes:
    * datadir moved to /data/yutong/datasets (matches run_sisa_cifar.sh;
      required for the cifar10 baseline to load).
    * rho_lr=1e2 pulled into COMMON_ARGS (was only in LIPSCHITZ_EXTRA_ARGS).
      run_sisa_cifar.sh uses rho_lr=1e2 for the SISA baseline; without this
      the cifar10 baseline would have been run with the entry point's
      default.
    * New wandb_project = paper-lipschitz-label1-cifar-extra to isolate
      these runs from legacy projects.
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_lipschitz_label1_cifar_extra")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]

# Concurrent runs per GPU. simple-cnn batch=64 is tiny on H100.
MAX_PARALLEL_PER_GPU = 8

SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

# Lipschitz estimator (fixed to ema; running_min lost the earlier pilot).
ESTIMATOR = "ema"
LIPSCHITZ_WINDOW_SIZE = "20"

# Only textbook_sc here; "none" and "inverse" were eliminated in prior pilots.
ETA_U_DECAY = "textbook_sc"

ONLINE_ENTRY = "experiment_sisa_practise_online.py"
ORIGINAL_ENTRY = "experiment_sisa_practise_wandb.py"

# Common args shared by both the Lipschitz online entry and the SISA baseline
# entry. datadir and rho_lr match run_sisa_cifar.sh so the cifar10 baseline
# loads correctly. rho_lr=1e2 is the SISA baseline default per the same file.
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
    "datadir": "/data/yutong/datasets",
    "logdir": "./logs/",
    "noise": "0",
    "sample": "1",
    "sigma_lr": "${sigma_lr}",
    "rho_lr": "1e2",
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": "paper-lipschitz-label1-cifar-extra",
}

# Extra args only for the Lipschitz online entry. rho_lr now lives in
# COMMON_ARGS (was here redundantly before); leaving it out on purpose so
# COMMON_ARGS remains the single source of truth.
LIPSCHITZ_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal_lipschitz",
    "sigma_min": "1e-6",
    "sigma_max": "1e6",
    "eta_u": "0.05",
    "eta_u_decay": ETA_U_DECAY,
    "G_clip": "5.0",
    "lipschitz_estimator": ESTIMATOR,
    "lipschitz_window_size": LIPSCHITZ_WINDOW_SIZE,
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
}

# Two disjoint job specs. See CHANGE LOG for rationale.
JOB_SPECS = [
    {
        "spec_id": "original_cifar",
        "entry": ORIGINAL_ENTRY,
        "extra_args": {},  # baseline needs no adaptive-sigma args
        "cases": [
            {"case_name": "cifar10_label1_n10", "dataset": "cifar10",
             "partition": "noniid-#label1", "model": "simple-cnn"},
            {"case_name": "cifar10_label2_n10", "dataset": "cifar10",
             "partition": "noniid-#label2", "model": "simple-cnn"},
            {"case_name": "cifar10_label3_n10", "dataset": "cifar10",
             "partition": "noniid-#label3", "model": "simple-cnn"},
        ],
        "seeds": [0, 1, 2],
        "tag": lambda sigma_lr: f"original_sig{sigma_lr}",
    },
    {
        "spec_id": "lipschitz_label1_extra",
        "entry": ONLINE_ENTRY,
        "extra_args": LIPSCHITZ_EXTRA_ARGS,
        "cases": [
            {"case_name": "mnist_label1_n10", "dataset": "mnist",
             "partition": "noniid-#label1", "model": "simple-cnn"},
            {"case_name": "fmnist_label1_n10", "dataset": "fmnist",
             "partition": "noniid-#label1", "model": "simple-cnn"},
            {"case_name": "cifar10_label1_n10", "dataset": "cifar10",
             "partition": "noniid-#label1", "model": "simple-cnn"},
        ],
        "seeds": [3, 4, 5, 6, 7, 8, 9],
        "tag": lambda sigma_lr: f"lipschitz_decay{ETA_U_DECAY}_sig{sigma_lr}",
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
    run_name = f"{case['dataset']}_{tag}_seed${{seed}}"
    return group, run_name


def build_command(spec: dict, case: dict, tag: str, cuda_device: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "partition": case["partition"],
    })
    args.update(spec["extra_args"])

    wandb_group, wandb_run_name = build_wandb_names(case, tag)
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {spec['entry']} \\"]
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

    # Collect all (spec, case, sigma, seed) jobs across specs.
    jobs = []
    for spec in JOB_SPECS:
        for case in spec["cases"]:
            for slr in SIGMA_LR_VALUES:
                for seed in spec["seeds"]:
                    tag = spec["tag"](slr)
                    script_name = f"{case['case_name']}_{tag}_seed{seed}.sh"
                    jobs.append((spec, case, slr, seed, tag, script_name))

    # Round-robin GPU assignment (one script = one seed).
    generated_scripts = []
    for idx, (spec, case, slr, seed, tag, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            spec, case, sigma_lr=slr, seed=seed, tag=tag, cuda_device=gpu
        )
        script_path.write_text(script_text, encoding="utf-8")
        make_executable(script_path)
        generated_scripts.append((script_path, gpu, spec["spec_id"]))
        print(f"Generated: {script_path}  [GPU {gpu}]  ({spec['spec_id']})")

    total = len(generated_scripts)
    spec_counts = {}
    for _, _, sid in generated_scripts:
        spec_counts[sid] = spec_counts.get(sid, 0) + 1
    print(f"\nGenerated {total} single-seed scripts across {len(CUDA_DEVICES)} GPUs.")
    for sid, n in spec_counts.items():
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
        futs = [ex.submit(run_one, sp, gpu) for sp, gpu, _ in generated_scripts]
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
