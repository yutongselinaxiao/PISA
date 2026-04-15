import stat
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("generated_sisa_exact_admm_runs")
LOG_DIR = OUTPUT_DIR / "logs"

# physical GPU id
CUDA_DEVICE = "1"

# SIGMA_LR = "1e2"
SEEDS = [0, 1, 2]

# Sweep over initial sigma for per-client runs
SIGMA_LR_VALUES = ["1e2", "1e3", "1e4"]

# Use the exact ADMM file for both methods
EXACT_ADMM_ENTRY = "experiment_sisa_practise_admm.py"    # adaptive
ORIGINAL_ENTRY = "experiment_sisa_practise_online.py"   # fixed
# ORIGINAL_ENTRY = "experiment_sisa_practise.py"
ORIGINAL_WANDB_ENTRY = "experiment_sisa_practise_wandb.py"    # original
PERCLIENT_ENTRY = "experiment_sisa_practise_perclient_adaptive_sigma.py"

# EXPERIMENT_TAG = f"exact_admm_shared_sigma_pilot_initsig_{SIGMA_LR}_4_7"

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
    "wandb_project": "sisa-exact-admm",
}

FIXED_EXTRA_ARGS = {
    "sigma_mode": "fixed",
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
    "epochs": "3"
}

HEURISTIC_EXTRA_ARGS = {
    "sigma_mode": "heuristic",
    "sigma_mu": "10.0",
    "sigma_tau": "2.0",
    "sigma_kmax": "50",
}

ORIGINAL_EXTRA_ARGS = {
    # "sigma_lr": "1e3",
    "rho_lr": "1e2",
    "comm_round": "500",
}

TASK_AWARE_EXTRA_ARGS = {
    "sigma_mode": "online_task_aware",
    "sigma_min": "1e-6",
    "sigma_max": "1e4",
    "eta_u": "0.05",
    "G_clip": "5.0",
    "eps": "1e-12",
    "sigma_update_freq": "1",
    "sigma_ema_beta": "0.9",
    "epochs": "3",
    "task_lambda": "${task_lambda}",
}

TASK_LAMBDA_VALUES = ["0.1", "1.0", "10.0"]

PERCLIENT_CONVBAL_ARGS = {
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
    "G_clip": "5.0",
}

# Start small; expand later
CASES = [
    {"case_name": "mnist_label3_n10", "dataset": "mnist", "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "fmnist_label3_n10", "dataset": "fmnist", "partition": "noniid-#label3", "model": "simple-cnn"},
    {"case_name": "mnist_label1_n10", "dataset": "mnist", "partition": "noniid-#label1", "model": "simple-cnn"},
    {"case_name": "fmnist_label1_n10", "dataset": "fmnist", "partition": "noniid-#label1", "model": "simple-cnn"},
    # {"case_name": "cifar10_label3_n10", "dataset": "cifar10", "partition": "noniid-#label3", "model": "simple-cnn"},
    # {"case_name": "mnist_label1_n10", "dataset": "mnist", "partition": "noniid-#label1", "model": "vgg"},
    # {"case_name": "fmnist_label1_n10", "dataset": "fmnist", "partition": "noniid-#label1", "model": "vgg"},
    # {"case_name": "cifar10_label3_n10", "dataset": "cifar10", "partition": "noniid-#label3", "model": "vgg"},
    # {"case_name": "mnist_label1_n10", "dataset": "mnist", "partition": "noniid-#label1", "model": "resnet"},
    # {"case_name": "fmnist_label1_n10", "dataset": "fmnist", "partition": "noniid-#label1", "model": "resnet"},
    # {"case_name": "cifar10_label3_n10", "dataset": "cifar10", "partition": "noniid-#label3", "model": "resnet"},
]

METHODS = [
    # {
    #     "method_name": "fixed",
    #     "entry": ORIGINAL_ENTRY,
    #     "extra_args": FIXED_EXTRA_ARGS,
    # },
    # {
    #     "method_name": "adaptive",
    #     "entry": EXACT_ADMM_ENTRY,
    #     "extra_args": ADAPTIVE_EXTRA_ARGS,
    #     "sweep_sigma": True,
    # },
    {
        "method_name": "task_aware",
        "entry": EXACT_ADMM_ENTRY,
        "extra_args": TASK_AWARE_EXTRA_ARGS,
        "sweep_sigma": True,
        "sweep_task_lambda": True,
    },
    # {
    #     "method_name": "heuristic",
    #     "entry": EXACT_ADMM_ENTRY,
    #     "extra_args": HEURISTIC_EXTRA_ARGS,
    # },
    # {
    #     "method_name": "original",
    #     "entry": ORIGINAL_WANDB_ENTRY,
    #     "extra_args": ORIGINAL_EXTRA_ARGS,
    # },
    # {
    #     "method_name": "perclient_convbal",
    #     "entry": PERCLIENT_ENTRY,
    #     "extra_args": PERCLIENT_CONVBAL_ARGS,
    #     "sweep_sigma": True,
    # },
]

# safer for first pass
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
    tag = f"exact_admm_shared_sigma_pilot_initsig_{sigma_lr_val}_4_7"
    # if task_lambda_val is not None:
    #     tag += f"_lam_{task_lambda_val}"
    return tag


def build_wandb_names(case: dict, method_name: str, tag: str = None):
    t = tag if tag is not None else EXPERIMENT_TAG
    group = f"{case['case_name']}-sisa-{t}"
    if method_name == "fixed":
        run_name = f"{case['dataset']}_sig${{sigma_lr}}_fixed_{t}_seed${{seed}}"
    elif method_name == "adaptive":
        run_name = f"{case['dataset']}_sig${{sigma_lr}}_convbal_sharedsigma_{t}_seed${{seed}}_updated_epochs_3"
    elif method_name == "heuristic":
        run_name = f"{case['dataset']}_sig${{sigma_lr}}_heuristic_{t}_seed${{seed}}_updated"
    elif method_name == "original":
        run_name = f"{case['dataset']}_original_{t}_seed${{seed}}"
    elif method_name == "task_aware":
        run_name = f"{case['dataset']}_sig${{sigma_lr}}_task_aware_lam${{task_lambda}}_{t}_seed${{seed}}"
    elif method_name == "perclient_convbal":
        run_name = f"{case['dataset']}_sig${{sigma_lr}}_perclient_convbal_stabilized_{t}_seed${{seed}}"
    else:
        raise ValueError(f"Unknown method_name: {method_name}")
    return group, run_name


def build_command_template(case: dict, method: dict, tag: str = None) -> str:
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


def build_script_text(case: dict, method: dict, sigma_lr: str = None,
                      task_lambda: str = None, tag: str = None) -> str:
    cmd = build_command_template(case, method, tag=tag)
    slr = sigma_lr if sigma_lr is not None else SIGMA_LR
    header_lines = [
        "#!/bin/bash",
        "",
        "set -e",
        "",
        f"sigma_lr={slr}",
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

            if method.get("sweep_sigma", False):
                for slr in SIGMA_LR_VALUES:
                    if sweep_tl:
                        for tl in TASK_LAMBDA_VALUES:
                            tag = make_experiment_tag(slr, tl)
                            script_name = f"{case['case_name']}_{method['method_name']}_{tag}.sh"
                            script_path = OUTPUT_DIR / script_name
                            script_text = build_script_text(case, method, sigma_lr=slr, task_lambda=tl, tag=tag)
                            script_path.write_text(script_text, encoding="utf-8")
                            make_executable(script_path)
                            generated_scripts.append(script_path)
                            print(f"Generated: {script_path}")
                    else:
                        tag = make_experiment_tag(slr)
                        script_name = f"{case['case_name']}_{method['method_name']}_{tag}.sh"
                        script_path = OUTPUT_DIR / script_name
                        script_text = build_script_text(case, method, sigma_lr=slr, tag=tag)
                        script_path.write_text(script_text, encoding="utf-8")
                        make_executable(script_path)
                        generated_scripts.append(script_path)
                        print(f"Generated: {script_path}")
            else:
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