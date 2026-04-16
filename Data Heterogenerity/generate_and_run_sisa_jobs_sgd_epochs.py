import stat
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_sgd_epochs_runs")
LOG_DIR = OUTPUT_DIR / "logs"

# physical GPU ids to distribute work across
CUDA_DEVICES = ["0", "1", "2", "3"]

SEEDS = [0, 1, 2]

# Sweep over initial sigma
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

# Sweep over local epochs
EPOCHS_VALUES = ["3", "10"]

# Sweep over local learning rate
LR_VALUES = ["0.001", "0.01"]

EXACT_ADMM_ENTRY = "experiment_sisa_practise_admm.py"

COMMON_ARGS = {
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
    "l2_lambda": "5e-3",
    "init_seed": "${seed}",
    "optimizer": "sgd",
    "use_wandb": "true",
    "wandb_project": "sisa-exact-admm-sgd-epochs",
}

ADAPTIVE_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal",
    "sigma_min": "1e-6",
    "sigma_max": "1e4",
    "eta_u": "0.05",
    "G_clip": "5.0",
    "eps": "1e-12",
    "sigma_update_freq": "1",
    "sigma_ema_beta": "0.9",
}

# Start small; expand later
CASES = [
    {"case_name": "mnist_label3_n10", "dataset": "mnist", "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "fmnist_label3_n10", "dataset": "fmnist", "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "mnist_label1_n10", "dataset": "mnist", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "fmnist_label1_n10", "dataset": "fmnist", "partition": "noniid-#label1", "model": "simple-cnn"},
]

FIXED_EXTRA_ARGS = {
    "sigma_mode": "fixed",
}

METHODS = [
    {
        "method_name": "sgd_adaptive",
        "entry": EXACT_ADMM_ENTRY,
        "extra_args": ADAPTIVE_EXTRA_ARGS,
        "sweep_sigma": True,
        "sweep_epochs": True,
        "sweep_lr": True,
    },
    {
        "method_name": "sgd_fixed",
        "entry": EXACT_ADMM_ENTRY,
        "extra_args": FIXED_EXTRA_ARGS,
        "sweep_sigma": True,
        "sweep_epochs": True,
        "sweep_lr": True,
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


def make_experiment_tag(sigma_lr_val: str, epochs_val: str, lr_val: str) -> str:
    return f"sgd_ep{epochs_val}_lr{lr_val}_sig{sigma_lr_val}_4_16"


def build_wandb_names(case: dict, method_name: str, tag: str):
    group = f"{case['case_name']}-sisa-{tag}"
    run_name = f"{case['dataset']}_sig${{sigma_lr}}_{method_name}_{tag}_seed${{seed}}"
    return group, run_name


def build_command_template(case: dict, method: dict, tag: str, cuda_device: str = "0",
                           epochs: str = "1", lr: str = "0.001") -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "alg": "sisa",
        "partition": case["partition"],
        "epochs": epochs,
        "lr": lr,
    })
    args.update(method["extra_args"])

    wandb_group, wandb_run_name = build_wandb_names(case, method["method_name"], tag=tag)
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={cuda_device} python {method['entry']} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(case: dict, method: dict, sigma_lr: str, tag: str,
                      cuda_device: str = "0", epochs: str = "1", lr: str = "0.001") -> str:
    cmd = build_command_template(case, method, tag=tag, cuda_device=cuda_device,
                                 epochs=epochs, lr=lr)
    return "\n".join([
        "#!/bin/bash",
        "",
        "set -e",
        "",
        f"sigma_lr={sigma_lr}",
        "",
        "for seed in 0 1 2",
        "do",
        cmd,
        "done",
        "",
    ])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all job specs first, then assign GPUs round-robin
    jobs = []

    for case in CASES:
        for method in METHODS:
            sigma_values = SIGMA_LR_VALUES if method.get("sweep_sigma", False) else ["1e2"]
            epochs_values = EPOCHS_VALUES if method.get("sweep_epochs", False) else ["1"]
            lr_values = LR_VALUES if method.get("sweep_lr", False) else ["0.001"]

            for slr in sigma_values:
                for ep in epochs_values:
                    for lr in lr_values:
                        tag = make_experiment_tag(slr, ep, lr)
                        script_name = f"{case['case_name']}_{method['method_name']}_{tag}.sh"
                        jobs.append((case, method, slr, ep, lr, tag, script_name))

    # Round-robin GPU assignment
    generated_scripts = []
    for idx, (case, method, slr, ep, lr, tag, script_name) in enumerate(jobs):
        gpu = CUDA_DEVICES[idx % len(CUDA_DEVICES)]
        script_path = OUTPUT_DIR / script_name
        script_text = build_script_text(
            case, method, sigma_lr=slr, tag=tag, cuda_device=gpu, epochs=ep, lr=lr
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
