"""Paper experiments: main accuracy table + hyperparameter sensitivity plot.

Covers 5 methods:
  - admm_adaptive  : our method (online_convex_bal), sweeps sigma_0
  - admm_fixed     : ADMM with fixed sigma, sweeps sigma
  - fedprox        : FedProx, sweeps mu
  - fedavg         : single config
  - scaffold       : single config

admm_adaptive / admm_fixed / fedprox sweeps double as the sensitivity plot.
Best of each sweep + fedavg + scaffold fill the main accuracy table.
"""

import stat
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("generated_paper_experiment_runs")
LOG_DIR = OUTPUT_DIR / "logs"

CUDA_DEVICES = ["0", "1", "2", "3"]
SEEDS = [0, 1, 2]

# Sensitivity sweep ranges
SIGMA_VALUES = ["1e1", "1e2", "1e3", "1e4", "1e5"]   # for ADMM fixed/adaptive
MU_VALUES = ["1e-3", "1e-2", "1e-1", "1.0", "10.0"]  # for FedProx

ENTRY = "experiment_sisa_practise_admm.py"

# Shared across every run
COMMON_ARGS = {
    "batch-size": "64",
    "n_parties": "10",
    "rho": "0.9",
    "comm_round": "500",
    "beta": "0.5",
    "device": "cuda:0",
    "datadir": "/dataMeR2/yutong/datasets",
    "logdir": "./logs/",
    "noise": "0",
    "sample": "1",
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "optimizer": "sgd",
    "lr": "0.01",
    "epochs": "3",
    "use_wandb": "true",
    "wandb_project": "paper-baselines",
}

ADMM_ADAPTIVE_EXTRA = {
    "alg": "sisa",
    "sigma_mode": "online_convex_bal",
    "sigma_lr": "${hp}",
    "sigma_min": "1e-6",
    "sigma_max": "1e6",
    "eta_u": "0.05",
    "G_clip": "5.0",
    "eps": "1e-12",
    "sigma_update_freq": "1",
    "sigma_ema_beta": "0.9",
    "mu": "0.01",
}

ADMM_FIXED_EXTRA = {
    "alg": "sisa",
    "sigma_mode": "fixed",
    "sigma_lr": "${hp}",
    "mu": "0.01",
}

FEDPROX_EXTRA = {
    "alg": "fedprox",
    "mu": "${hp}",
}

FEDAVG_EXTRA = {
    "alg": "fedavg",
    "mu": "0.0",
}

SCAFFOLD_EXTRA = {
    "alg": "scaffold",
    "mu": "0.0",
}

CASES = [
    {"case_name": "mnist_label3_n10",  "dataset": "mnist",  "partition": "noniid-#label3", "model": "simple-cnn"},
    # {"case_name": "fmnist_label3_n10", "dataset": "fmnist", "partition": "noniid-#label3", "model": "simple-cnn"},
    # {"case_name": "mnist_label1_n10",  "dataset": "mnist",  "partition": "noniid-#label1", "model": "simple-cnn"},
    # {"case_name": "fmnist_label1_n10", "dataset": "fmnist", "partition": "noniid-#label1", "model": "simple-cnn"},
]

METHODS = [
    {"name": "admm_adaptive", "extra": ADMM_ADAPTIVE_EXTRA, "hp_name": "sigma",  "hp_values": SIGMA_VALUES},
    {"name": "admm_fixed",    "extra": ADMM_FIXED_EXTRA,    "hp_name": "sigma",  "hp_values": SIGMA_VALUES},
    {"name": "fedprox",       "extra": FEDPROX_EXTRA,       "hp_name": "mu",     "hp_values": MU_VALUES},
    {"name": "fedavg",        "extra": FEDAVG_EXTRA,        "hp_name": None,     "hp_values": [None]},
    {"name": "scaffold",      "extra": SCAFFOLD_EXTRA,      "hp_name": None,     "hp_values": [None]},
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


def make_tag(method_name: str, hp_name: str | None, hp_value: str | None) -> str:
    if hp_name is None:
        return f"{method_name}"
    return f"{method_name}_{hp_name}{hp_value}"


def build_wandb_names(case: dict, method_name: str, hp_name: str | None, tag: str):
    group = f"{case['case_name']}-{method_name}"
    if hp_name is None:
        run_name = f"{case['dataset']}_{method_name}_seed${{seed}}"
    else:
        run_name = f"{case['dataset']}_{method_name}_{hp_name}${{hp}}_seed${{seed}}"
    return group, run_name


def build_command(case: dict, method: dict, tag: str, cuda_device: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "partition": case["partition"],
    })
    args.update(method["extra"])

    wandb_group, wandb_run_name = build_wandb_names(
        case, method["name"], method["hp_name"], tag
    )
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {ENTRY} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script(case: dict, method: dict, hp_value: str | None,
                 tag: str, cuda_device: str) -> str:
    cmd = build_command(case, method, tag=tag, cuda_device=cuda_device)
    header = [
        "#!/bin/bash",
        "",
        "set -e",
        "",
    ]
    if hp_value is not None:
        header.append(f"hp={hp_value}")
        header.append("")
    header += [
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
        for method in METHODS:
            for hp_value in method["hp_values"]:
                tag = make_tag(method["name"], method["hp_name"], hp_value)
                script_name = f"{case['case_name']}_{tag}.sh"
                jobs.append((case, method, hp_value, tag, script_name))

    generated_scripts = []
    for idx, (case, method, hp_value, tag, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script(case, method, hp_value, tag, gpu)
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
