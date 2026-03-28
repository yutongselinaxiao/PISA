import stat
import subprocess
from pathlib import Path

# New save folder for harder short pilot runs
OUTPUT_DIR = Path("generated_sisa_runs_500_n10_label1_sigma0_1000_rho_0.8_v7")
LOG_DIR = OUTPUT_DIR / "logs"

# All generated scripts will use physical GPU 3.
# Inside python we still use --device='cuda:0' because CUDA_VISIBLE_DEVICES remaps it.
CUDA_DEVICE = "1"

SIGMA_LR = "1e3"
SEEDS = [0, 1, 2]

FIXED_ENTRY = "experiment_sisa_practise_online.py"
ADAPTIVE_ENTRY = "experiment_sisa_practise_perclient_adaptive_sigma.py"

# Tag used in wandb names
EXPERIMENT_TAG = "500_n10_label1_sigma0_1000_rho_0.8_v7"

COMMON_ARGS = {
    "lr": "0.001",
    "batch-size": "64",
    "epochs": "1",
    "n_parties": "10",
    "mu": "0.01",
    "rho": "0.8",
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
    "wandb_project": "sisa-adaptive-sigma",
}

FIXED_EXTRA_ARGS = {
    "sigma_mode": "fixed",
}

ADAPTIVE_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal",
    "sigma_update_freq": "1",
    "eta_u": "1e-1",
    "sigma_max_delta": "0.2",
    "sigma_max_delta_min": "0.01",
    "sigma_ema_beta": "0.9",
    "sigma_blend": "0.8",
    "sigma_blend_min": "0.05",
    "sigma_stabilize_start": "80",
    "sigma_deadband": "0.1",
    "sigma_min": "1e-6",
    "sigma_max": "1e4",
    "eps": "1e-12",
}

# ADAPTIVE_EXTRA_ARGS = { "sigma_mode": "online_convex_bal", "sigma_update_freq": "1", "eta_u": "1e-1", "sigma_max_delta": "0.2", "sigma_max_delta_min": "0.01", "sigma_ema_beta": "0.9", "sigma_blend": "0.8", "sigma_blend_min": "0.05", "sigma_stabilize_start": "80", "sigma_deadband": "0.1", "sigma_min": "1e-6", "sigma_max": "1e6", "eps": "1e-12", }

HYBRID_EXTRA_ARGS = {
    "sigma_mode": "online_hybrid",
    "sigma_update_freq": "10",
    "eta_u": "1e-1",
    "sigma_max_delta": "0.2",
    "sigma_max_delta_min": "0.01",
    "sigma_ema_beta": "0.9",
    "sigma_blend": "0.8",
    "sigma_blend_min": "0.05",
    "sigma_stabilize_start": "80",
    "sigma_deadband": "0.1",
    "sigma_min": "1e-6",
    "sigma_max": "1e4",
    "eps": "1e-12",
    "hybrid_alpha": "1.0",
    "hybrid_lambda0": "2.0",
    "hybrid_tau_mag": "0.1",
}

# Harder stress-test cases: all 1-label with 30 clients
CASES = [
    {"case_name": "mnist_1label_n10", "dataset": "mnist", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "fmnist_1label_n10", "dataset": "fmnist", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "cifar10_1label_n10", "dataset": "cifar10", "partition": "noniid-#label1", "model": "simple-cnn"},
]

METHODS = [
    {
        "method_name": "fixed",
        "entry": FIXED_ENTRY,
        "extra_args": FIXED_EXTRA_ARGS,
    },
    {
        "method_name": "adaptive",
        "entry": ADAPTIVE_ENTRY,
        "extra_args": ADAPTIVE_EXTRA_ARGS,
    },
    {
        "method_name": "hybrid",
        "entry": ADAPTIVE_ENTRY,
        "extra_args": HYBRID_EXTRA_ARGS,
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


def build_wandb_names(case: dict, method_name: str):
    if method_name == "fixed":
        group = f"{case['case_name']}-sisa-{EXPERIMENT_TAG}"
        run_name = (
            f"{case['dataset']}_sigma_${{sigma_lr}}_fixed_"
            f"{EXPERIMENT_TAG}_seed${{seed}}"
        )
    elif method_name == "adaptive":
        group = f"{case['case_name']}-sisa-{EXPERIMENT_TAG}"
        run_name = (
            f"{case['dataset']}_sigma_${{sigma_lr}}_adaptive_"
            f"convbal_upd10_eta5e-2_delta5e-2_ema95_blend03_"
            f"{EXPERIMENT_TAG}_seed${{seed}}"
        )
    elif method_name == "hybrid":
        group = f"{case['case_name']}-sisa-{EXPERIMENT_TAG}"
        run_name = (
            f"{case['dataset']}_sigma_${{sigma_lr}}_hybrid_"
            f"upd10_eta5e-2_delta5e-2_ema95_blend03_"
            f"a1_lam2_tau01_{EXPERIMENT_TAG}_seed${{seed}}"
        )
    else:
        raise ValueError(f"Unknown method_name: {method_name}")
    return group, run_name


def build_command_template(case: dict, method: dict) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "alg": "sisa",
        "partition": case["partition"],
    })
    args.update(method["extra_args"])

    wandb_group, wandb_run_name = build_wandb_names(case, method["method_name"])
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python {method['entry']} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(case: dict, method: dict) -> str:
    cmd = build_command_template(case, method)
    return "\n".join([
        "#!/bin/bash",
        "",
        "set -e",
        "",
        f"sigma_lr={SIGMA_LR}",
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

    generated_scripts = []

    for case in CASES:
        for method in METHODS:
            script_name = f"{case['case_name']}_{method['method_name']}_{EXPERIMENT_TAG}.sh"
            script_path = OUTPUT_DIR / script_name
            script_text = build_script_text(case, method)
            script_path.write_text(script_text, encoding="utf-8")
            make_executable(script_path)
            generated_scripts.append(script_path)
            print(f"Generated: {script_path}")

    if not RUN_AFTER_GENERATION:
        print("\nGeneration complete. Not executing scripts.")
        return

    print(f"\nLaunching all scripts in parallel on physical GPU {CUDA_DEVICE}...\n")

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