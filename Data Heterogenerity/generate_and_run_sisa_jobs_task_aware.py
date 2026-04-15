import stat
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_task_aware_runs")
LOG_DIR = OUTPUT_DIR / "logs"

# physical GPU id
CUDA_DEVICE = "0"

SEEDS = [0, 1, 2]

# Sweep over initial sigma
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

# Sweep over task_lambda
TASK_LAMBDA_VALUES = ["0.1", "1.0", "10.0"]

ONLINE_ENTRY = "experiment_sisa_practise_online.py"

COMMON_ARGS = {
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
    "wandb_project": "sisa-task-aware-sigma",
}

# Baseline: existing online_convex_bal on linearized ADMM
CONVBAL_EXTRA_ARGS = {
    "sigma_mode": "online_convex_bal",
    "sigma_min": "1e-6",
    "sigma_max": "1e6",
    "eta_u": "0.05",
    "G_clip": "5.0",
    "rho_lr": "1e2",
}

# New: task-aware online loss
TASK_AWARE_EXTRA_ARGS = {
    "sigma_mode": "online_task_aware",
    "sigma_min": "1e-6",
    "sigma_max": "1e6",
    "eta_u": "0.05",
    "G_clip": "5.0",
    "task_lambda": "${task_lambda}",
    "rho_lr": "1e2",
}

CASES = [
    {"case_name": "mnist_label3_n10", "dataset": "mnist", "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "fmnist_label3_n10", "dataset": "fmnist", "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "mnist_label1_n10", "dataset": "mnist", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "fmnist_label1_n10", "dataset": "fmnist", "partition": "noniid-#label1", "model": "simple-cnn"},
]

METHODS = [
    {
        "method_name": "convbal",
        "entry": ONLINE_ENTRY,
        "extra_args": CONVBAL_EXTRA_ARGS,
        "sweep_sigma": True,
    },
    {
        "method_name": "task_aware",
        "entry": ONLINE_ENTRY,
        "extra_args": TASK_AWARE_EXTRA_ARGS,
        "sweep_sigma": True,
        "sweep_task_lambda": True,
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


def make_experiment_tag(sigma_lr_val: str, task_lambda_val: str = None) -> str:
    tag = f"task_aware_initsig_{sigma_lr_val}"
    if task_lambda_val is not None:
        tag += f"_lam_{task_lambda_val}"
    return tag


def build_wandb_names(case: dict, method_name: str, tag: str):
    group = f"{case['case_name']}-sisa-{tag}"
    if method_name == "convbal":
        run_name = f"{case['dataset']}_sig${{sigma_lr}}_convbal_{tag}_seed${{seed}}"
    elif method_name == "task_aware":
        run_name = f"{case['dataset']}_sig${{sigma_lr}}_task_aware_lam${{task_lambda}}_{tag}_seed${{seed}}"
    else:
        raise ValueError(f"Unknown method_name: {method_name}")
    return group, run_name


def build_command_template(case: dict, method: dict, tag: str) -> str:
    args = {}
    args.update(COMMON_ARGS)
    args.update({
        "model": case["model"],
        "dataset": case["dataset"],
        "alg": "sisa",
        "partition": case["partition"],
    })
    args.update(method["extra_args"])

    wandb_group, wandb_run_name = build_wandb_names(case, method["method_name"], tag=tag)
    args["wandb_group"] = wandb_group
    args["wandb_run_name"] = wandb_run_name

    lines = [f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python {method['entry']} \\"]
    items = list(args.items())
    for i, (k, v) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        lines.append(f"    {format_arg(k, v)}{suffix}")
    return "\n".join(lines)


def build_script_text(case: dict, method: dict, sigma_lr: str,
                      task_lambda: str = None, tag: str = None) -> str:
    cmd = build_command_template(case, method, tag=tag)
    header_lines = [
        "#!/bin/bash",
        "",
        "set -e",
        "",
        f"sigma_lr={sigma_lr}",
    ]
    if task_lambda is not None:
        header_lines.append(f"task_lambda={task_lambda}")
    header_lines += [
        "",
        "for seed in 0 1 2",
        "do",
        cmd,
        "done",
        "",
    ]
    return "\n".join(header_lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    generated_scripts = []

    for case in CASES:
        for method in METHODS:
            sweep_tl = method.get("sweep_task_lambda", False)

            for slr in SIGMA_LR_VALUES:
                if sweep_tl:
                    # sweep over task_lambda for task_aware method
                    for tl in TASK_LAMBDA_VALUES:
                        tag = make_experiment_tag(slr, tl)
                        script_name = f"{case['case_name']}_{method['method_name']}_{tag}.sh"
                        script_path = OUTPUT_DIR / script_name
                        script_text = build_script_text(
                            case, method, sigma_lr=slr, task_lambda=tl, tag=tag
                        )
                        script_path.write_text(script_text, encoding="utf-8")
                        make_executable(script_path)
                        generated_scripts.append(script_path)
                        print(f"Generated: {script_path}")
                else:
                    tag = make_experiment_tag(slr)
                    script_name = f"{case['case_name']}_{method['method_name']}_{tag}.sh"
                    script_path = OUTPUT_DIR / script_name
                    script_text = build_script_text(
                        case, method, sigma_lr=slr, tag=tag
                    )
                    script_path.write_text(script_text, encoding="utf-8")
                    make_executable(script_path)
                    generated_scripts.append(script_path)
                    print(f"Generated: {script_path}")

    total = len(generated_scripts)
    print(f"\nGenerated {total} scripts total.")

    if not RUN_AFTER_GENERATION:
        print("Not executing scripts.")
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
