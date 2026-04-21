"""Generator for the per-client Lipschitz-floor pilot (2026-04-20).

Compares two residual definitions inside per_client.py's new
online_convex_bal_lipschitz_{oldres,newres} modes:

- oldres: delta_y = ||W_global - W_global_prev||   (shared global step)
- newres: delta_y = ||W_n       - W_n_prev||       (per-client iterate change)

Both modes share the same update function (ported verbatim from online.py) and
a per-client EMA-smoothed BB Lipschitz estimate with hard projection
u_i >= log L_hat_i. Step size follows the textbook strongly-convex schedule
eta_k = 1/(mu*k) with mu=2; `--eta_u` is ignored.

Cases: {mnist, fmnist} x {label1, label2, label3} x sigma_0 in {1e2, 1e3, 1e4}
       x sigma_mode in {oldres, newres}. 3 seeds per script, run sequentially.

The residual-axis split tests whether per-client dual captures client
heterogeneity better than the shared global step when sigma is per-client.
The sigma_0 sweep tests robustness of the hard Lipschitz projection.
"""

import stat
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_perclient_lipschitz")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["4", "5", "6", "7"]

SEEDS = [0, 1, 2]

SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

SIGMA_MODES = [
    "online_convex_bal_lipschitz_oldres",
    "online_convex_bal_lipschitz_newres",
]

ESTIMATOR = "ema"
LIPSCHITZ_WINDOW_SIZE = "20"
ETA_U_DECAY = "textbook_sc"

ONLINE_ENTRY = "experiment_sisa_practise_perclient_adaptive_sigma.py"

WANDB_PROJECT = "paper-perclient-lipschitz"

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
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "use_wandb": "true",
    "wandb_project": WANDB_PROJECT,
}

LIPSCHITZ_EXTRA_ARGS = {
    "sigma_mode": "${sigma_mode}",
    "sigma_min": "1e-6",
    "sigma_max": "1e6",
    "eta_u": "0.05",
    "eta_u_decay": ETA_U_DECAY,
    "G_clip": "5.0",
    "rho_lr": "1e2",
    "lipschitz_estimator": ESTIMATOR,
    "lipschitz_window_size": LIPSCHITZ_WINDOW_SIZE,
    "lipschitz_ema_beta": "0.9",
    "lipschitz_min_dz": "1e-6",
    "lipschitz_max": "1e8",
    "sigma_update_freq": "1",
}

CASES = [
    {"case_name": "mnist_label1_n10",  "dataset": "mnist",  "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "mnist_label2_n10",  "dataset": "mnist",  "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "mnist_label3_n10",  "dataset": "mnist",  "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "fmnist_label1_n10", "dataset": "fmnist", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "fmnist_label2_n10", "dataset": "fmnist", "partition": "noniid-#label2", "model": "simple-cnn"},
    {"case_name": "fmnist_label3_n10", "dataset": "fmnist", "partition": "noniid-#label3", "model": "simple-cnn"},
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


def short_mode_tag(mode: str) -> str:
    if mode.endswith("_oldres"):
        return "oldres"
    if mode.endswith("_newres"):
        return "newres"
    return mode


def make_experiment_tag(sigma_lr_val: str, mode: str) -> str:
    return f"perclient_lip_{short_mode_tag(mode)}_sig{sigma_lr_val}"


def build_wandb_names(case: dict, tag: str):
    group = f"{case['case_name']}-{tag}"
    run_name = f"{case['dataset']}_{tag}_seed${{seed}}"
    return group, run_name


def build_command(case: dict, tag: str, cuda_device: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "partition": case["partition"],
    })
    args.update(LIPSCHITZ_EXTRA_ARGS)

    wandb_group, wandb_run_name = build_wandb_names(case, tag)
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {ONLINE_ENTRY} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(case: dict, sigma_lr: str, mode: str,
                      tag: str, cuda_device: str) -> str:
    cmd = build_command(case, tag=tag, cuda_device=cuda_device)
    header = [
        "#!/bin/bash",
        "",
        "set -e",
        "",
        f"sigma_lr={sigma_lr}",
        f"sigma_mode={mode}",
        "",
        "for seed in 0 1 2",
        "do",
        cmd,
        "done",
        "",
    ]
    return "\n".join(header)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for case in CASES:
        for mode in SIGMA_MODES:
            for slr in SIGMA_LR_VALUES:
                tag = make_experiment_tag(slr, mode)
                jobs.append((case, slr, mode, tag))

    generated_scripts = []
    for idx, (case, slr, mode, tag) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_name = f"{case['case_name']}_{tag}.sh"
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            case, sigma_lr=slr, mode=mode, tag=tag, cuda_device=gpu
        )
        script_path.write_text(script_text, encoding="utf-8")
        make_executable(script_path)
        generated_scripts.append(script_path)
        print(f"Generated: {script_path}  [GPU {gpu}]")

    total = len(generated_scripts)
    print(f"\nGenerated {total} scripts across {len(CUDA_DEVICES)} GPUs.")
    print(f"Each script runs {len(SEEDS)} seeds sequentially = {total * len(SEEDS)} total runs.")

    if not RUN_AFTER_GENERATION:
        print("Not executing scripts.")
        return

    print(f"\nLaunching all scripts in parallel across GPUs {CUDA_DEVICES}...\n")

    processes = []
    for script_path in generated_scripts:
        log_path = LOG_DIR / f"{script_path.stem}.log"
        print(f"Launching: {script_path} -> {log_path}")
        log_file = open(log_path, "w")
        p = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((script_path, log_path, log_file, p))

    print("\nAll scripts launched.\n")

    failed = []
    for script_path, log_path, log_file, p in processes:
        ret = p.wait()
        log_file.close()
        if ret == 0:
            print(f"Finished: {script_path}")
        else:
            print(f"FAILED: {script_path} with exit code {ret}. See log: {log_path}")
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
