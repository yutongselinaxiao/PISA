"""Improved-baselines FL sweep — strong adaptive-σ comparisons.

Currently runs ONE method:
  improved_heuristic — Boyd residual-balance heuristic with EMA β=0.9 smoothing
                       and k_max=50 cutoff, the published-paper-faithful
                       implementation. Entry: experiment_sisa_practise_admm.py.

Two more methods from `/home/yutong/online_admm/online_admm_applications/`
that we discussed earlier are NOT YET RUNNABLE here because they live in
JAX/NumPy and need a port into experiment_sisa_practise_admm.py:

  spectral_aadmm    — Xu-Figueiredo-Goldstein 2017 (the canonical SOTA
                      adaptive-ρ baseline). Needs a context-tracking port
                      (intermediate dual λ_hat, final dual λ, A x, B z).
  residual_balance_normalized — Wohlberg 2017 (heuristic on residuals
                      scaled by ADMM stopping thresholds). Smaller port
                      than spectral_aadmm.

When those are ported, add them as additional JOB_SPECS below.

Cells: mnist/fmnist/cifar10 × label1/2/3 × σ_0 ∈ {1e2, 1e3, 1e4} × 3 seeds.
Total: 1 method × 9 cells × 3 σ × 3 seeds = 81 runs (will scale up to
243 once the other two methods are ported).

Wandb project: paper-canonical-fl (same as the four primary methods, so
the dashboard can compare all baselines side-by-side; the
`improved_heuristic_*` run_name prefix keeps these visually separable
from the four-method runs in the wandb UI).
"""

import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_improved_baselines")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3", "4", "5", "6", "7"]
MAX_PARALLEL_PER_GPU = 8

SEEDS = [0, 1, 2]
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

ADMM_ENTRY = "experiment_sisa_practise_admm.py"
# When NormalizedResidualBalancing / SpectralAADMM are ported, the entry
# stays the same — they'd be additional --sigma_mode choices in _admm.py.

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
    "wandb_project": "paper-canonical-fl",
}

IMPROVED_HEURISTIC_EXTRA_ARGS = {
    "sigma_mode": "heuristic",
    "sigma_min": "1e-6",
    "sigma_max": "1e8",
    "sigma_mu": "10.0",
    "sigma_tau": "2.0",
    "sigma_ema_beta": "0.9",
    "sigma_kmax": "50",
    "sigma_update_freq": "1",
}

# When porting NormalizedResidualBalancing to _admm.py, expect roughly:
#   "sigma_mode": "residual_balance_normalized",
#   "sigma_mu": "10.0", "sigma_tau": "2.0",
#   "primal_threshold": "1.0",  # ADMM stopping threshold; tunable
#   "dual_threshold":   "1.0",
#
# When porting SpectralAADMM (more involved), expect:
#   "sigma_mode": "spectral_aadmm",
#   "spectral_update_period": "2",
#   "spectral_correlation_threshold": "0.2",
#   plus _admm.py needs to track λ_hat, λ, h=Ax, g=Bz across iterations.

JOB_SPECS = [
    {
        "spec_id": "improved_heuristic",
        "entry": ADMM_ENTRY,
        "extra_args": IMPROVED_HEURISTIC_EXTRA_ARGS,
        "tag": lambda sigma_lr: f"improved_heuristic_mu10_tau2_kmax50_emab0p9_sig{sigma_lr}",
    },
    # TODO once ported into _admm.py:
    # {
    #     "spec_id": "residual_balance_normalized",
    #     "entry": ADMM_ENTRY,
    #     "extra_args": {**NORMALIZED_BALANCE_ARGS},
    #     "tag": lambda sigma_lr: f"normalized_balance_mu10_tau2_sig{sigma_lr}",
    # },
    # {
    #     "spec_id": "spectral_aadmm",
    #     "entry": ADMM_ENTRY,
    #     "extra_args": {**SPECTRAL_AADMM_ARGS},
    #     "tag": lambda sigma_lr: f"spectral_aadmm_xu2017_sig{sigma_lr}",
    # },
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
    grp, name = build_wandb_names(case, tag)
    args["wandb_group"] = grp
    args["wandb_run_name"] = name

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {spec['entry']} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(spec, case, sigma_lr, seed, tag, cuda_device):
    cmd = build_command(spec, case, tag, cuda_device)
    return "\n".join([
        "#!/bin/bash", "", "set -e", "",
        f"sigma_lr={sigma_lr}", f"seed={seed}", "",
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
    by_spec = {}
    for _, _, sid in generated:
        by_spec[sid] = by_spec.get(sid, 0) + 1
    print(f"\nGenerated {total} single-seed scripts.")
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
