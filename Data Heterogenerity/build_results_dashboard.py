#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None
else:
    if not hasattr(wandb, "Api"):
        wandb = None


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
OUT_JSON = ROOT / "results_dashboard_data.json"
OUT_HTML = ROOT / "results_dashboard.html"

ARG_PREFIX = "experiment_arguments-"
ARG_SUFFIX = ".json"
LOG_PREFIX = "experiment_log-"
LOG_SUFFIX = ".log"

ACC_RE = re.compile(r"Global Model Test accuracy:\s*([0-9.]+)")
ROUND_RE = re.compile(r"ADMM round\s+(\d+)")
SIGMA_RE = re.compile(r"^\S+\s+\S+\s+INFO\s+>> sigma_lr:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
WANDB_URL_RE = re.compile(r"https://wandb\.ai/([^/\s]+)/([^/\s]+)")
WANDB_RUN_URL_RE = re.compile(r"https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([^/\s]+)")
WANDB_RUN_ID_RE = re.compile(r"wandb:\s+setting up run\s+([a-zA-Z0-9]+)")
WANDB_SYNCING_NAME_RE = re.compile(r"wandb:\s+Syncing run\s+(.+)")
DISCOVER_LOG_GLOBS = ("**/*.log",)
TEST_ACC_KEYS = ("test/acc", "test_acc", "test_accuracy", "accuracy/test", "test/accuracy")
PROJECT_PRESETS: dict[str, list[str]] = {
    "sgd": [
        "sisa-exact-admm-sgd-epochs-4-22",
        "sisa-exact-admm-warmstart",
    ],
    "lipschitz": [
        "paper-lipschitz-estimator",
        "paper-perclient-lipschitz",
    ],
    "classic": [
        "sisa-exact-admm",
        "sisa-task-aware-sigma",
    ],
}
EXCLUDED_RUN_NAME_MARKERS_BY_PROJECT: dict[str, tuple[str, ...]] = {
    "sisa-exact-admm": ("perclient", "no_ema"),
}
# Per-project config filter: a run is kept only if its config matches ALL
# fields listed here. Used to carve out a subset (e.g. only adam_warmstart
# runs from sisa-exact-admm, ignoring the other 247 sgd/baseline/fixed runs).
REQUIRED_CONFIG_BY_PROJECT: dict[str, dict[str, Any]] = {
    "sisa-exact-admm": {"optimizer": "adam_warmstart"},
}
SGD_EXACT_ADMM_PROJECT = "sisa-exact-admm-sgd-epochs-4-22"
WARMSTART_PROJECT = "sisa-exact-admm-warmstart"
LIPSCHITZ_MAIN_PROJECT = "paper-lipschitz-estimator"
DIRECT_SISA_PROJECTS = {"sisa-adaptive-sigma", "sisa-adaptive-sigma-debug"}
EXACT_ADMM_PROJECT = "sisa-exact-admm"
COMPARABLE_DELTA = 0.01


def parse_namespace_line(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("Empty argument payload")
    start = stripped.find("Namespace(")
    if start == -1:
        raise ValueError(f"Unexpected namespace payload: {stripped[:80]}")

    depth = 0
    in_string = False
    string_char = ""
    escape = False
    end = None
    for idx, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == string_char:
                in_string = False
        else:
            if char in {"'", '"'}:
                in_string = True
                string_char = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
    if end is None:
        raise ValueError("Could not find balanced Namespace(...) payload")

    text = stripped[start:end]
    if not text.startswith("Namespace(") or not text.endswith(")"):
        raise ValueError(f"Unexpected namespace payload: {text[:80]}")
    inner = text[len("Namespace(") : -1]
    expr = ast.parse(f"f({inner})", mode="eval")
    call = expr.body
    if not isinstance(call, ast.Call):
        raise ValueError("Failed to parse Namespace payload")
    parsed: dict[str, Any] = {}
    for kw in call.keywords:
        parsed[kw.arg] = ast.literal_eval(kw.value)
    return parsed


def stem_from_name(name: str, prefix: str, suffix: str) -> str:
    return name[len(prefix) : -len(suffix)]


def sci_label(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if value == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(float(value)))))
    mantissa = float(value) / (10**exp)
    if abs(mantissa - 1.0) < 1e-9:
        return f"1e{exp}"
    return f"{mantissa:.1f}e{exp}"


def compact_partition(value: Any) -> str:
    if not value:
        return "unknown"
    return str(value).replace("noniid-#label", "label")


def detect_family(config: dict[str, Any]) -> str:
    project = str(config.get("wandb_project") or "")
    rho_lr = config.get("rho_lr")
    epochs = config.get("epochs")
    if project in {"paper-lipschitz-estimator", "paper-lipschitz-floor", "paper-perclient-lipschitz"}:
        return "lipschitz-sisa"
    if project == "sisa-exact-admm":
        if rho_lr == 100 and epochs == 1 and config.get("sigma_mode") == "fixed":
            return "sisa-baseline"
        return "exact-admm"
    if "linearized" in project:
        return "linearized-sisa"
    return project or "unknown"


def extract_numeric_metric(value: Any, *, prefer: str = "last") -> float | None:
    # Dict-like first. W&B's SummarySubDict is NOT a dict or Mapping subclass
    # — it just implements dict-like methods. Its __getattr__ proxies to
    # __getitem__ and raises KeyError for missing keys, so the naïve
    # `hasattr(value, "item")` path blows up on summary values shaped like
    # {'last': 0.88, 'max': 0.89}. Use class-level duck-typing to detect it
    # without triggering __getattr__ on the instance.
    cls = type(value)
    if (
        not isinstance(value, (str, bytes))
        and hasattr(cls, "keys")
        and hasattr(cls, "__getitem__")
    ):
        try:
            available_keys = set(value.keys())
        except Exception:
            available_keys = None
        if available_keys is not None:
            for subkey in (prefer, "last", "max", "value", "mean"):
                if subkey in available_keys:
                    try:
                        unwrapped = extract_numeric_metric(value[subkey], prefer=prefer)
                    except Exception:
                        continue
                    if unwrapped is not None:
                        return unwrapped
            return None
    if hasattr(cls, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        metric = float(value)
        return metric if math.isfinite(metric) else None
    if isinstance(value, str):
        try:
            metric = float(value.strip())
        except ValueError:
            return None
        return metric if math.isfinite(metric) else None
    return None


def first_numeric_metric(values: Iterable[Any]) -> float | None:
    for value in values:
        metric = extract_numeric_metric(value)
        if metric is not None:
            return metric
    return None


def summary_metric(summary: dict[str, Any], keys: Iterable[str]) -> float | None:
    metric, _key = summary_metric_with_key(summary, keys)
    return metric


def summary_metric_with_key(summary: dict[str, Any], keys: Iterable[str]) -> tuple[float | None, str | None]:
    candidates: list[Any] = []
    for key in keys:
        for candidate_key in (key, f"{key}.last", f"{key}.max", f"{key}_last", f"{key}_max"):
            metric = extract_numeric_metric(summary.get(candidate_key))
            if metric is not None:
                return metric, candidate_key
            candidates.append(summary.get(candidate_key))
    return first_numeric_metric(candidates), None


def row_metric(row: Any, key: str) -> float | None:
    if not isinstance(row, dict):
        return None
    return summary_metric(row, (key,))


def rows_from_history(history: Any) -> list[dict[str, Any]]:
    if history is None:
        return []
    if hasattr(history, "to_dict"):
        try:
            records = history.to_dict("records")
            return [row for row in records if isinstance(row, dict)]
        except (TypeError, ValueError):
            return []
    if isinstance(history, list):
        return [row for row in history if isinstance(row, dict)]
    try:
        return [row for row in history if isinstance(row, dict)]
    except TypeError:
        return []


def fetch_history_values_for_key(run: Any, key: str) -> tuple[list[float], str | None]:
    values: list[float] = []
    errors: list[str] = []

    try:
        for row in run.scan_history(keys=[key], page_size=1000):
            metric = row_metric(row, key)
            if metric is not None:
                values.append(metric)
    except Exception as exc:
        errors.append(f"scan_history({key}): {exc}")

    if values:
        return values, None

    for pandas_mode in (False, True):
        try:
            history = run.history(keys=[key], samples=10000, pandas=pandas_mode)
            for row in rows_from_history(history):
                metric = row_metric(row, key)
                if metric is not None:
                    values.append(metric)
        except Exception as exc:
            errors.append(f"history({key}, pandas={pandas_mode}): {exc}")
        if values:
            return values, None

    return [], "; ".join(errors[:2]) if errors else None


def fetch_history_test_metrics(run: Any) -> tuple[float | None, float | None, int, str | None, str | None]:
    errors: list[str] = []
    for key in TEST_ACC_KEYS:
        values, error = fetch_history_values_for_key(run, key)
        if values:
            return values[-1], max(values), len(values), f"history:{key}", None
        if error:
            errors.append(error)
    return None, None, 0, None, "; ".join(errors[:2]) if errors else None


def metric_like_summary_keys(summary: dict[str, Any], limit: int = 12) -> list[str]:
    needles = ("acc", "accuracy", "test")
    keys = sorted(str(key) for key in summary if any(needle in str(key).lower() for needle in needles))
    return keys[:limit]


def infer_wandb_entity(search_root: Path) -> str | None:
    entities: Counter[str] = Counter()
    for pattern in DISCOVER_LOG_GLOBS:
        for path in search_root.glob(pattern):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for entity, _project in WANDB_URL_RE.findall(text):
                entities[entity] += 1
    return entities.most_common(1)[0][0] if entities else None


def expand_project_selection(
    explicit_projects: list[str] | None,
    preset_names: list[str] | None,
) -> list[str] | None:
    selected: list[str] = []
    for preset_name in preset_names or []:
        selected.extend(PROJECT_PRESETS.get(preset_name, []))
    selected.extend(explicit_projects or [])
    deduped = sorted(dict.fromkeys(project for project in selected if project))
    return deduped or None


def should_exclude_run(run: dict[str, Any]) -> bool:
    project = str(run.get("project") or "")
    run_name = str(run.get("run_name") or "").lower()
    for marker in EXCLUDED_RUN_NAME_MARKERS_BY_PROJECT.get(project, ()):
        if marker in run_name:
            return True
    required = REQUIRED_CONFIG_BY_PROJECT.get(project)
    if required:
        config = run.get("config") or {}
        for key, expected in required.items():
            actual = config.get(key)
            if actual is None:
                actual = run.get(key)
            if actual != expected:
                return True
    return False


def filter_excluded_runs(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    filtered = [run for run in runs if not should_exclude_run(run)]
    return filtered, len(runs) - len(filtered)


def normalize_remote_status(state: str | None, last_test_acc: float | None) -> str:
    normalized = (state or "").lower()
    if normalized == "finished":
        return "finished" if last_test_acc is not None else "no-metric"
    if normalized in {"failed", "crashed", "killed", "cancelled"}:
        return "failed"
    if normalized in {"running", "pending", "preempting", "queued"}:
        return "partial"
    return "partial" if last_test_acc is not None else "no-metric"


def parse_log_metrics(log_path: Path, comm_round: int | None) -> dict[str, Any]:
    text = log_path.read_text(errors="ignore")
    accs = [float(match) for match in ACC_RE.findall(text)]
    rounds = [int(match) for match in ROUND_RE.findall(text)]
    sigmas = [float(match) for match in SIGMA_RE.findall(text)]
    run_urls = WANDB_RUN_URL_RE.findall(text)
    run_ids = WANDB_RUN_ID_RE.findall(text)
    syncing_names = WANDB_SYNCING_NAME_RE.findall(text)
    max_round = max(rounds) if rounds else None
    complete = bool(
        comm_round
        and accs
        and max_round is not None
        and max_round >= int(comm_round) - 1
        and len(accs) >= int(comm_round)
    )
    failed = "No space left on device" in text or "Traceback (most recent call last)" in text
    if failed:
        status = "failed"
    elif complete:
        status = "finished"
    elif accs:
        status = "partial"
    else:
        status = "no-metric"
    return {
        "status": status,
        "has_disk_error": "No space left on device" in text,
        "last_test_acc": accs[-1] if accs else None,
        "best_test_acc": max(accs) if accs else None,
        "num_test_points": len(accs),
        "max_round": max_round,
        "last_sigma": sigmas[-1] if sigmas else None,
        "log_size_bytes": log_path.stat().st_size,
        "updated_at": datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "wandb_url": f"https://wandb.ai/{run_urls[-1][0]}/{run_urls[-1][1]}/runs/{run_urls[-1][2]}" if run_urls else None,
        "wandb_entity": run_urls[-1][0] if run_urls else None,
        "wandb_project": run_urls[-1][1] if run_urls else None,
        "wandb_id": run_urls[-1][2] if run_urls else (run_ids[-1] if run_ids else None),
        "wandb_log_run_name": syncing_names[-1].strip() if syncing_names else None,
    }


def iter_log_dirs() -> list[Path]:
    seen: set[Path] = set()
    log_dirs: list[Path] = []
    for path in sorted(ROOT.glob("**/logs")):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        log_dirs.append(path)
    return log_dirs


def load_local_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for logs_dir in iter_log_dirs():
        arg_files = {stem_from_name(path.name, ARG_PREFIX, ARG_SUFFIX): path for path in logs_dir.glob(f"{ARG_PREFIX}*{ARG_SUFFIX}")}
        log_files = {stem_from_name(path.name, LOG_PREFIX, LOG_SUFFIX): path for path in logs_dir.glob(f"{LOG_PREFIX}*{LOG_SUFFIX}")}

        for stem, arg_path in sorted(arg_files.items()):
            config = parse_namespace_line(arg_path.read_text())
            log_path = log_files.get(stem)
            metrics = parse_log_metrics(log_path, config.get("comm_round")) if log_path else {
                "status": "missing-log",
                "has_disk_error": False,
                "last_test_acc": None,
                "best_test_acc": None,
                "num_test_points": 0,
                "max_round": None,
                "last_sigma": None,
                "log_size_bytes": 0,
                "updated_at": None,
            }
            run = {
                "stem": stem,
                "timestamp": stem,
                "dataset": config.get("dataset"),
                "partition": compact_partition(config.get("partition")),
                "project": config.get("wandb_project"),
                "group": config.get("wandb_group"),
                "run_name": config.get("wandb_run_name"),
                "optimizer": config.get("optimizer"),
                "sigma_mode": config.get("sigma_mode", "fixed"),
                "epochs": config.get("epochs"),
                "comm_round": config.get("comm_round"),
                "sigma_init": config.get("sigma_lr"),
                "sigma_init_label": sci_label(config.get("sigma_lr")),
                "rho_lr": config.get("rho_lr"),
                "seed": config.get("init_seed"),
                "model": config.get("model"),
                "n_parties": config.get("n_parties"),
                "family": detect_family(config),
                "data_source": "local",
                "config": config,
                "arg_path": str(arg_path),
                "log_path": str(log_path) if log_path else None,
                "source_log_dir": str(logs_dir),
                "wandb_id": None,
                "wandb_url": None,
                "wandb_entity": None,
                "wandb_state": None,
                "wandb_log_run_name": None,
            }
            run.update(metrics)
            if run.get("wandb_project"):
                run["project"] = run["wandb_project"]
            if run.get("wandb_log_run_name") and not run.get("run_name"):
                run["run_name"] = run["wandb_log_run_name"]
            runs.append(run)
    return merge_runs(runs, [])


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def fetch_wandb_runs(
    entity: str,
    projects: Iterable[str],
    timeout: int = 30,
    run_names_by_project: dict[str, list[str]] | None = None,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    if wandb is None:
        raise RuntimeError("wandb is not installed in the current interpreter")

    api = wandb.Api(timeout=timeout)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for project in sorted({project for project in projects if project}):
        api_path = f"{entity}/{project}"
        try:
            requested_names = sorted(set((run_names_by_project or {}).get(project, [])))
            if progress:
                if requested_names:
                    print(f"[wandb] Fetching {api_path} with {len(requested_names)} run-name filters")
                else:
                    print(f"[wandb] Fetching full project {api_path}")
            iterables = []
            if requested_names:
                for names in chunked(requested_names, 50):
                    iterables.append(api.runs(api_path, filters={"display_name": {"$in": names}}, per_page=100))
            else:
                iterables.append(api.runs(api_path, per_page=200, order="-created_at"))

            for project_runs in iterables:
                project_count = 0
                for run in project_runs:
                    config = dict(run.config)
                    summary = dict(run.summary)
                    last_test_acc, summary_metric_key = summary_metric_with_key(summary, TEST_ACC_KEYS)
                    best_test_acc = None
                    for _key in TEST_ACC_KEYS:
                        raw = summary.get(_key)
                        if raw is None:
                            raw = summary.get(f"{_key}.max") or summary.get(f"{_key}_max")
                        if raw is None:
                            continue
                        best_test_acc = extract_numeric_metric(raw, prefer="max")
                        if best_test_acc is not None:
                            break
                    if best_test_acc is None:
                        best_test_acc = last_test_acc
                    num_test_points = None
                    metric_source = f"summary:{summary_metric_key}" if last_test_acc is not None else None
                    metric_error = None
                    if last_test_acc is None:
                        if progress:
                            print(f"[wandb] {api_path}: summary has no test accuracy for {run.name}; scanning history")
                        history_last, history_best, history_count, history_source, history_error = fetch_history_test_metrics(run)
                        if history_last is not None:
                            last_test_acc = history_last
                            best_test_acc = history_best or history_last
                            num_test_points = history_count
                            metric_source = history_source
                        else:
                            metric_error = history_error
                            if progress:
                                keys = ", ".join(metric_like_summary_keys(summary)) or "no acc/test-like summary keys"
                                print(f"[wandb] {api_path}: no history test accuracy for {run.name}; summary keys: {keys}")
                    updated_at = getattr(run, "heartbeat_at", None) or getattr(run, "updated_at", None) or getattr(run, "created_at", None)
                    row = {
                        "stem": None,
                        "timestamp": getattr(run, "created_at", None),
                        "dataset": config.get("dataset"),
                        "partition": compact_partition(config.get("partition")),
                        "project": project,
                        "group": getattr(run, "group", None),
                        "run_name": run.name,
                        "optimizer": config.get("optimizer"),
                        "sigma_mode": config.get("sigma_mode", "fixed"),
                        "epochs": config.get("epochs"),
                        "comm_round": config.get("comm_round"),
                        "sigma_init": config.get("sigma_lr"),
                        "sigma_init_label": sci_label(config.get("sigma_lr")),
                        "rho_lr": config.get("rho_lr"),
                        "seed": config.get("init_seed"),
                        "model": config.get("model"),
                        "n_parties": config.get("n_parties"),
                        "family": detect_family({**config, "wandb_project": project}),
                        "data_source": "wandb",
                        "config": config,
                        "arg_path": None,
                        "log_path": None,
                        "status": normalize_remote_status(getattr(run, "state", None), last_test_acc),
                        "has_disk_error": False,
                        "last_test_acc": last_test_acc,
                        "best_test_acc": best_test_acc,
                        "num_test_points": num_test_points,
                        "max_round": extract_numeric_metric(summary.get("_step")),
                        "last_sigma": extract_numeric_metric(summary.get("sigma_lr")),
                        "log_size_bytes": None,
                        "updated_at": updated_at,
                        "wandb_id": getattr(run, "id", None),
                        "wandb_url": getattr(run, "url", None),
                        "wandb_entity": entity,
                        "wandb_state": getattr(run, "state", None),
                        "wandb_metric_source": metric_source,
                        "wandb_metric_error": metric_error,
                        "wandb_summary_metric_keys": metric_like_summary_keys(summary) if last_test_acc is None else [],
                    }
                    rows.append(row)
                    project_count += 1
                    if progress and project_count % 50 == 0:
                        print(f"[wandb] {api_path}: loaded {project_count} runs")
                if progress:
                    print(f"[wandb] {api_path}: loaded {project_count} runs total for current iterator")
        except Exception as exc:
            errors.append(f"{project}: {exc}")
    return rows, errors


def merge_runs(local_runs: list[dict[str, Any]], remote_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    fallback_index: dict[tuple[Any, ...], tuple[Any, ...]] = {}

    def primary_key(run: dict[str, Any]) -> tuple[Any, ...]:
        wandb_id = run.get("wandb_id")
        if wandb_id:
            return ("wandb_id", wandb_id)
        wandb_url = run.get("wandb_url")
        if wandb_url:
            return ("wandb_url", wandb_url)
        return (
            "logical",
            run.get("project"),
            run.get("run_name"),
            run.get("seed"),
            run.get("dataset"),
            run.get("partition"),
        )

    def logical_key(run: dict[str, Any]) -> tuple[Any, ...]:
        return (
            run.get("project"),
            run.get("run_name"),
            run.get("seed"),
            run.get("dataset"),
            run.get("partition"),
        )

    for run in local_runs:
        key = primary_key(run)
        merged[key] = dict(run)
        fallback_index[logical_key(run)] = key

    for remote in remote_runs:
        key = primary_key(remote)
        existing = merged.get(key)
        if existing is None:
            existing = merged.get(fallback_index.get(logical_key(remote), ()))
        if existing is None:
            merged[key] = dict(remote)
            fallback_index[logical_key(remote)] = key
            continue

        combined = dict(existing)
        combined["data_source"] = "merged"
        combined["wandb_id"] = remote.get("wandb_id")
        combined["wandb_url"] = remote.get("wandb_url")
        combined["wandb_entity"] = remote.get("wandb_entity")
        combined["wandb_state"] = remote.get("wandb_state")
        combined["wandb_metric_source"] = remote.get("wandb_metric_source") or existing.get("wandb_metric_source")
        combined["wandb_metric_error"] = remote.get("wandb_metric_error") or existing.get("wandb_metric_error")
        combined["wandb_summary_metric_keys"] = remote.get("wandb_summary_metric_keys") or existing.get("wandb_summary_metric_keys")
        combined["group"] = remote.get("group") or existing.get("group")
        combined["timestamp"] = remote.get("timestamp") or existing.get("timestamp")
        combined["updated_at"] = remote.get("updated_at") or existing.get("updated_at")

        for field in ("status", "last_test_acc", "best_test_acc", "max_round", "last_sigma"):
            if remote.get(field) is not None:
                combined[field] = remote[field]

        if remote.get("status") == "finished":
            combined["status"] = "finished"
        elif existing.get("status") not in {"finished", "failed"} and remote.get("status"):
            combined["status"] = remote["status"]

        existing_key = primary_key(existing)
        if existing_key != key and existing_key in merged:
            del merged[existing_key]
        merged[key] = combined
        fallback_index[logical_key(combined)] = key

    rows = list(merged.values())
    rows.sort(key=lambda row: (str(row.get("timestamp") or ""), str(row.get("project") or ""), str(row.get("run_name") or "")))
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def build_dashboard(
    out_json: Path,
    out_html: Path,
    source: str,
    wandb_entity: str | None,
    wandb_projects: list[str] | None,
    wandb_timeout: int,
    prefer_local_run_names: bool,
    progress: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runs, source_metadata = load_runs(
        source=source,
        wandb_entity=wandb_entity,
        wandb_projects=wandb_projects,
        wandb_timeout=wandb_timeout,
        prefer_local_run_names=prefer_local_run_names,
        progress=progress,
    )
    runs, excluded_count = filter_excluded_runs(runs)
    source_metadata = dict(source_metadata)
    source_metadata["excluded_runs"] = excluded_count
    payload = build_payload(runs, source_metadata=source_metadata)
    atomic_write(out_json, json.dumps(payload, indent=2))
    atomic_write(out_html, render_html(payload))
    return source_metadata, runs


def load_runs(
    source: str,
    wandb_entity: str | None = None,
    wandb_projects: list[str] | None = None,
    wandb_timeout: int = 30,
    prefer_local_run_names: bool = True,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    local_runs = load_local_runs()
    metadata: dict[str, Any] = {
        "source_mode": source,
        "wandb_enabled": False,
        "wandb_entity": wandb_entity,
        "wandb_projects": sorted({run["project"] for run in local_runs if run.get("project")}),
        "wandb_error": None,
    }
    if source == "local":
        return local_runs, metadata

    projects = wandb_projects or metadata["wandb_projects"]
    entity = wandb_entity or infer_wandb_entity(ROOT)
    metadata["wandb_entity"] = entity
    metadata["wandb_projects"] = projects
    run_names_by_project: dict[str, list[str]] | None = None
    if source in {"auto", "wandb"} and prefer_local_run_names:
        grouped_names: dict[str, list[str]] = defaultdict(list)
        for run in local_runs:
            project = run.get("project")
            run_name = run.get("run_name")
            if project and run_name:
                grouped_names[project].append(run_name)
        run_names_by_project = dict(grouped_names)

    if not entity:
        if source == "wandb":
            raise RuntimeError("Could not infer wandb entity. Pass --wandb-entity explicitly.")
        metadata["wandb_error"] = "Could not infer wandb entity; using local data only."
        return local_runs, metadata

    try:
        remote_runs, errors = fetch_wandb_runs(
            entity=entity,
            projects=projects,
            timeout=wandb_timeout,
            run_names_by_project=run_names_by_project,
            progress=progress,
        )
        metadata["wandb_enabled"] = True
        metadata["wandb_run_count"] = len(remote_runs)
        metadata["wandb_project_errors"] = errors
        if errors:
            metadata["wandb_error"] = "; ".join(errors[:3])
        if source == "wandb":
            return remote_runs, metadata
        return merge_runs(local_runs, remote_runs), metadata
    except Exception as exc:
        if source == "wandb":
            raise
        metadata["wandb_error"] = str(exc)
        return local_runs, metadata


def numeric(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float))]


def build_group_summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["last_test_acc"] is None:
            continue
        key = (
            run["family"],
            run["project"],
            run["dataset"],
            run["partition"],
            run["optimizer"],
            run["sigma_mode"],
            run["epochs"],
            run["sigma_init_label"],
        )
        buckets[key].append(run)

    summaries: list[dict[str, Any]] = []
    for key, rows in buckets.items():
        scores = numeric([row["last_test_acc"] for row in rows])
        if not scores:
            continue
        summaries.append(
            {
                "family": key[0],
                "project": key[1],
                "dataset": key[2],
                "partition": key[3],
                "optimizer": key[4],
                "sigma_mode": key[5],
                "epochs": key[6],
                "sigma_init_label": key[7],
                "n_runs": len(rows),
                "mean_test_acc": statistics.fmean(scores),
                "min_test_acc": min(scores),
                "max_test_acc": max(scores),
                "spread": max(scores) - min(scores),
                "finished_runs": sum(1 for row in rows if row["status"] == "finished"),
                "partial_runs": sum(1 for row in rows if row["status"] == "partial"),
            }
        )

    summaries.sort(key=lambda row: (-row["mean_test_acc"], row["dataset"], row["partition"], row["project"]))
    return summaries


def build_best_cells(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run in runs:
        score = run["last_test_acc"]
        if score is None:
            continue
        key = (run["dataset"], run["partition"])
        prev = best.get(key)
        if prev is None or score > prev["last_test_acc"]:
            best[key] = run
    rows = []
    for (_, _), run in sorted(best.items()):
        rows.append(
            {
                "dataset": run["dataset"],
                "partition": run["partition"],
                "last_test_acc": run["last_test_acc"],
                "project": run["project"],
                "family": run["family"],
                "optimizer": run["optimizer"],
                "sigma_mode": run["sigma_mode"],
                "epochs": run["epochs"],
                "sigma_init_label": run["sigma_init_label"],
                "seed": run["seed"],
                "status": run["status"],
                "run_name": run["run_name"],
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["partition"]))
    return rows


def setup_signature(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        run.get("project"),
        run.get("family"),
        run.get("optimizer"),
        run.get("sigma_mode"),
        run.get("epochs"),
        run.get("sigma_init_label"),
    )


def setup_descriptor_from_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": run.get("project"),
        "family": run.get("family"),
        "optimizer": run.get("optimizer"),
        "sigma_mode": run.get("sigma_mode"),
        "epochs": run.get("epochs"),
        "sigma_init_label": run.get("sigma_init_label"),
    }


def build_individual_ranking(runs: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    ranked = [run for run in runs if run.get("last_test_acc") is not None]
    ranked.sort(
        key=lambda run: (
            -float(run["last_test_acc"]),
            str(run.get("dataset") or ""),
            str(run.get("partition") or ""),
            str(run.get("project") or ""),
            str(run.get("run_name") or ""),
        )
    )
    rows: list[dict[str, Any]] = []
    for rank, run in enumerate(ranked[:limit], start=1):
        rows.append(
            {
                "rank": rank,
                "dataset": run.get("dataset"),
                "partition": run.get("partition"),
                "last_test_acc": run.get("last_test_acc"),
                "best_test_acc": run.get("best_test_acc"),
                "seed": run.get("seed"),
                "status": run.get("status"),
                "run_name": run.get("run_name"),
                "wandb_url": run.get("wandb_url"),
                **setup_descriptor_from_run(run),
            }
        )
    return rows


def build_setup_combo_leaders(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaders: list[dict[str, Any]] = []
    by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_cell[(row["dataset"], row["partition"])].append(row)

    for (dataset, partition), rows in sorted(by_cell.items()):
        ranked_rows = sorted(
            rows,
            key=lambda row: (-row["mean_test_acc"], -row["max_test_acc"], -row["n_runs"], str(row["project"] or "")),
        )
        top = ranked_rows[0]
        leaders.append(
            {
                "dataset": dataset,
                "partition": partition,
                "mean_test_acc": top["mean_test_acc"],
                "max_test_acc": top["max_test_acc"],
                "n_runs": top["n_runs"],
                "finished_runs": top["finished_runs"],
                "partial_runs": top["partial_runs"],
                **setup_descriptor_from_run(top),
            }
        )
    leaders.sort(key=lambda row: (-row["mean_test_acc"], row["dataset"], row["partition"]))
    for rank, row in enumerate(leaders, start=1):
        row["rank"] = rank
    return leaders


def build_combined_setup_ranking(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[setup_signature(row)].append(row)

    combined: list[dict[str, Any]] = []
    for rows in grouped.values():
        combo_scores = [float(row["mean_test_acc"]) for row in rows]
        if not combo_scores:
            continue
        best_combo = max(rows, key=lambda row: row["mean_test_acc"])
        combined.append(
            {
                "project": rows[0]["project"],
                "family": rows[0]["family"],
                "optimizer": rows[0]["optimizer"],
                "sigma_mode": rows[0]["sigma_mode"],
                "epochs": rows[0]["epochs"],
                "sigma_init_label": rows[0]["sigma_init_label"],
                "combo_count": len(rows),
                "datasets_covered": len({row["dataset"] for row in rows}),
                "partitions_covered": len({row["partition"] for row in rows}),
                "overall_mean": statistics.fmean(combo_scores),
                "best_combo_mean": best_combo["mean_test_acc"],
                "best_combo_dataset": best_combo["dataset"],
                "best_combo_partition": best_combo["partition"],
                "total_runs": sum(int(row["n_runs"]) for row in rows),
                "finished_runs": sum(int(row["finished_runs"]) for row in rows),
                "partial_runs": sum(int(row["partial_runs"]) for row in rows),
            }
        )

    combined.sort(
        key=lambda row: (
            -row["overall_mean"],
            -row["combo_count"],
            -row["best_combo_mean"],
            str(row["project"] or ""),
        )
    )
    for rank, row in enumerate(combined, start=1):
        row["rank"] = rank
    return combined


def run_name_lower(run: dict[str, Any]) -> str:
    return str(run.get("run_name") or "").lower()


def config_value(run: dict[str, Any], key: str, default: Any = None) -> Any:
    return (run.get("config") or {}).get(key, default)


def scored_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run for run in runs if run.get("last_test_acc") is not None]


def summarize_scored_runs(runs: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    rows = scored_runs(runs)
    if not rows:
        return None
    scores = [float(run["last_test_acc"]) for run in rows]
    best_run = max(rows, key=lambda run: float(run["last_test_acc"]))
    return {
        "n_runs": len(rows),
        "mean_test_acc": statistics.fmean(scores),
        "max_test_acc": max(scores),
        "best_sigma": best_run.get("sigma_init_label"),
        "best_epochs": best_run.get("epochs"),
        "best_run_name": best_run.get("run_name"),
        "best_task_lambda": config_value(best_run, "task_lambda"),
    }


def choose_best_summary(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row.get("mean_test_acc") or float("-inf")),
            float(row.get("max_test_acc") or float("-inf")),
            int(row.get("n_runs") or 0),
        ),
    )


def comparison_label(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    if abs(delta) <= COMPARABLE_DELTA:
        return "comparable"
    return "better" if delta > 0 else "worse"


def epoch_help_label(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    if delta > 0.01:
        return "yes"
    if delta >= -0.005:
        return "mostly flat"
    return "no"


def sigma_order_key(label: Any) -> tuple[int, str]:
    order = {"1e3": 0, "1e2": 1, "1e4": 2}
    text = str(label or "")
    return (order.get(text, 99), text)


def is_sgd_convex_bal_run(run: dict[str, Any]) -> bool:
    return (
        run.get("project") == SGD_EXACT_ADMM_PROJECT
        and (run.get("sigma_mode") == "online_convex_bal" or "sgd_adaptive" in run_name_lower(run))
    )


def is_sgd_original_run(run: dict[str, Any]) -> bool:
    return run.get("project") == SGD_EXACT_ADMM_PROJECT and "sgd_original" in run_name_lower(run)


def is_sisa_baseline_run(run: dict[str, Any]) -> bool:
    """SISA baseline (no adaptive sigma) across both the SGD and Lipschitz
    paper projects. Cifar10 baselines live in the Lipschitz project; mnist/
    fmnist baselines live in both. Use for cross-project comparisons
    (sections 5 and 6). Matches via sigma_mode in {None, 'fixed'} with a
    fallback to run-name markers for older runs that set it differently."""
    project = run.get("project") or ""
    if project not in (SGD_EXACT_ADMM_PROJECT, LIPSCHITZ_MAIN_PROJECT):
        return False
    sigma_mode = run.get("sigma_mode")
    if sigma_mode in (None, "fixed", "None"):
        return True
    name = run_name_lower(run)
    return "sgd_original" in name or "original_sig" in name


def is_lipschitz_convex_bal_run(run: dict[str, Any]) -> bool:
    return run.get("project") == LIPSCHITZ_MAIN_PROJECT and "lipschitz" in str(run.get("sigma_mode") or "")


def is_lipschitz_original_run(run: dict[str, Any]) -> bool:
    return run.get("project") == LIPSCHITZ_MAIN_PROJECT and not is_lipschitz_convex_bal_run(run)


def is_lipschitz_textbook_sc_run(run: dict[str, Any]) -> bool:
    if run.get("project") != LIPSCHITZ_MAIN_PROJECT:
        return False
    config = run.get("config") or {}
    return (
        config.get("sigma_mode") == "online_convex_bal_lipschitz"
        and str(config.get("eta_u_decay") or "").lower() == "textbook_sc"
    )


def is_direct_sisa_convex_bal_run(run: dict[str, Any]) -> bool:
    return run.get("project") in DIRECT_SISA_PROJECTS and run.get("sigma_mode") == "online_convex_bal"


def is_exact_admm_adam_warmstart_run(run: dict[str, Any]) -> bool:
    return (
        run.get("project") == EXACT_ADMM_PROJECT
        and (str(run.get("optimizer") or "") == "adam_warmstart" or "adam_warmstart" in run_name_lower(run))
    )


def is_exact_admm_task_aware_run(run: dict[str, Any]) -> bool:
    return (
        run.get("project") == EXACT_ADMM_PROJECT
        and (str(run.get("sigma_mode") or "") == "online_task_aware" or "task_aware" in run_name_lower(run))
    )


def build_sgd_epoch_comparison_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    convex_runs = scored_runs(run for run in runs if is_sgd_convex_bal_run(run))
    original_runs = scored_runs(run for run in runs if is_sgd_original_run(run))

    original_by_cell: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in original_runs:
        original_by_cell[(run.get("dataset"), run.get("partition"))].append(run)

    convex_by_group: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in convex_runs:
        convex_by_group[(run.get("dataset"), run.get("partition"), run.get("epochs"))].append(run)

    rows: list[dict[str, Any]] = []
    for (dataset, partition, epochs), group_runs in sorted(convex_by_group.items()):
        sigma_candidates: list[dict[str, Any]] = []
        sigma_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for run in group_runs:
            sigma_groups[run.get("sigma_init_label")].append(run)
        for sigma, sigma_runs in sigma_groups.items():
            summary = summarize_scored_runs(sigma_runs)
            if summary is not None:
                sigma_candidates.append({"sigma_init_label": sigma, **summary})
        best_convex = choose_best_summary(sigma_candidates)
        original_sigma_candidates: list[dict[str, Any]] = []
        original_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for run in original_by_cell.get((dataset, partition), []):
            original_groups[run.get("sigma_init_label")].append(run)
        for sigma, sigma_runs in original_groups.items():
            summary = summarize_scored_runs(sigma_runs)
            if summary is not None:
                original_sigma_candidates.append({"sigma_init_label": sigma, **summary})
        best_original = choose_best_summary(original_sigma_candidates)
        delta = None if not best_convex or not best_original else best_convex["mean_test_acc"] - best_original["mean_test_acc"]
        rows.append(
            {
                "dataset": dataset,
                "partition": compact_partition(partition),
                "epochs": epochs,
                "best_sigma": None if not best_convex else best_convex["sigma_init_label"],
                "convex_mean": None if not best_convex else best_convex["mean_test_acc"],
                "original_best_sigma": None if not best_original else best_original["sigma_init_label"],
                "original_mean": None if not best_original else best_original["mean_test_acc"],
                "delta_vs_original": delta,
                "comparison": comparison_label(delta),
                "convex_runs": None if not best_convex else best_convex["n_runs"],
                "original_runs": None if not best_original else best_original["n_runs"],
            }
        )
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["partition"]), int(row["epochs"] or 0)))
    return rows


def build_sgd_sigma_comparison_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    convex_runs = scored_runs(run for run in runs if is_sgd_convex_bal_run(run))
    original_runs = scored_runs(run for run in runs if is_sgd_original_run(run))

    original_by_cell: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in original_runs:
        original_by_cell[(run.get("dataset"), run.get("partition"))].append(run)

    convex_by_group: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in convex_runs:
        convex_by_group[(run.get("dataset"), run.get("partition"), run.get("sigma_init_label"))].append(run)

    rows: list[dict[str, Any]] = []
    for (dataset, partition, sigma_label), group_runs in convex_by_group.items():
        epoch_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for run in group_runs:
            epoch_groups[run.get("epochs")].append(run)
        epoch_candidates: list[dict[str, Any]] = []
        for epochs, epoch_runs in epoch_groups.items():
            summary = summarize_scored_runs(epoch_runs)
            if summary is not None:
                epoch_candidates.append({"epochs": epochs, **summary})
        best_convex = choose_best_summary(epoch_candidates)

        original_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for run in original_by_cell.get((dataset, partition), []):
            original_groups[run.get("sigma_init_label")].append(run)
        original_candidates: list[dict[str, Any]] = []
        for original_sigma, sigma_runs in original_groups.items():
            summary = summarize_scored_runs(sigma_runs)
            if summary is not None:
                original_candidates.append({"sigma_init_label": original_sigma, **summary})
        best_original = choose_best_summary(original_candidates)
        delta = None if not best_convex or not best_original else best_convex["mean_test_acc"] - best_original["mean_test_acc"]

        rows.append(
            {
                "dataset": dataset,
                "partition": compact_partition(partition),
                "sigma_init_label": sigma_label,
                "best_epoch": None if not best_convex else best_convex["epochs"],
                "convex_mean": None if not best_convex else best_convex["mean_test_acc"],
                "original_best_sigma": None if not best_original else best_original["sigma_init_label"],
                "original_mean": None if not best_original else best_original["mean_test_acc"],
                "delta_vs_original": delta,
                "comparison": comparison_label(delta),
            }
        )
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["partition"]), sigma_order_key(row["sigma_init_label"])))
    return rows


def build_sgd_epoch_effect_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    convex_runs = scored_runs(run for run in runs if is_sgd_convex_bal_run(run))
    by_cell: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in convex_runs:
        by_cell[(run.get("dataset"), run.get("partition"))].append(run)

    target_epochs = [1, 3, 10]
    rows: list[dict[str, Any]] = []
    for (dataset, partition), cell_runs in sorted(by_cell.items()):
        row: dict[str, Any] = {"dataset": dataset, "partition": compact_partition(partition)}
        epoch_best_map: dict[int, dict[str, Any]] = {}
        for epoch in target_epochs:
            sigma_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for run in cell_runs:
                if int(run.get("epochs") or 0) == epoch:
                    sigma_groups[run.get("sigma_init_label")].append(run)
            candidates: list[dict[str, Any]] = []
            for sigma, sigma_runs in sigma_groups.items():
                summary = summarize_scored_runs(sigma_runs)
                if summary is not None:
                    candidates.append({"sigma_init_label": sigma, **summary})
            best_epoch = choose_best_summary(candidates)
            epoch_best_map[epoch] = best_epoch or {}
            row[f"ep{epoch}_sigma"] = None if not best_epoch else best_epoch["sigma_init_label"]
            row[f"ep{epoch}_mean"] = None if not best_epoch else best_epoch["mean_test_acc"]

        best_epoch_num = None
        best_epoch_score = None
        for epoch in target_epochs:
            score = row.get(f"ep{epoch}_mean")
            if score is not None and (best_epoch_score is None or score > best_epoch_score):
                best_epoch_num = epoch
                best_epoch_score = score
        delta_vs_ep1 = None if row.get("ep1_mean") is None or best_epoch_score is None else best_epoch_score - row["ep1_mean"]
        row["best_epoch"] = best_epoch_num
        row["delta_vs_ep1"] = delta_vs_ep1
        row["more_epochs_help"] = epoch_help_label(delta_vs_ep1)
        rows.append(row)
    return rows


def build_lipschitz_comparison_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lipschitz_runs = scored_runs(run for run in runs if is_lipschitz_convex_bal_run(run))
    # Non-Lipschitz convex-bal baseline: exact-ADMM SGD convex-bal at epochs=1
    # from the current -4-22 sweep.
    nonlip_runs = scored_runs(
        run for run in runs
        if is_sgd_convex_bal_run(run) and int(run.get("epochs") or 0) == 1
    )
    # Original SISA baseline across both the SGD sweep project and the
    # Lipschitz project. Cifar10 baselines only exist in the Lipschitz
    # project, so we can't restrict to the SGD project here.
    original_runs = scored_runs(run for run in runs if is_sisa_baseline_run(run))

    lipschitz_by_cell: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    nonlip_by_cell: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    original_by_cell: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)

    for run in lipschitz_runs:
        lipschitz_by_cell[(run.get("dataset"), run.get("partition"))].append(run)
    for run in nonlip_runs:
        nonlip_by_cell[(run.get("dataset"), run.get("partition"))].append(run)
    for run in original_runs:
        original_by_cell[(run.get("dataset"), run.get("partition"))].append(run)

    all_cells = sorted(set(lipschitz_by_cell) | set(nonlip_by_cell) | set(original_by_cell))
    rows: list[dict[str, Any]] = []
    for dataset, partition in all_cells:
        def best_by_sigma(cell_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
            sigma_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for run in cell_runs:
                sigma_groups[run.get("sigma_init_label")].append(run)
            candidates: list[dict[str, Any]] = []
            for sigma, sigma_runs in sigma_groups.items():
                summary = summarize_scored_runs(sigma_runs)
                if summary is not None:
                    candidates.append({"sigma_init_label": sigma, **summary})
            return choose_best_summary(candidates)

        best_lipschitz = best_by_sigma(lipschitz_by_cell.get((dataset, partition), []))
        best_nonlip = best_by_sigma(nonlip_by_cell.get((dataset, partition), []))
        best_original = best_by_sigma(original_by_cell.get((dataset, partition), []))
        if best_lipschitz is None:
            continue

        delta_nonlip = None if not best_lipschitz or not best_nonlip else best_lipschitz["mean_test_acc"] - best_nonlip["mean_test_acc"]
        delta_original = None if not best_lipschitz or not best_original else best_lipschitz["mean_test_acc"] - best_original["mean_test_acc"]

        rows.append(
            {
                "dataset": dataset,
                "partition": compact_partition(partition),
                "lipschitz_best_sigma": None if not best_lipschitz else best_lipschitz["sigma_init_label"],
                "lipschitz_mean": None if not best_lipschitz else best_lipschitz["mean_test_acc"],
                "nonlip_best_sigma": None if not best_nonlip else best_nonlip["sigma_init_label"],
                "nonlip_mean": None if not best_nonlip else best_nonlip["mean_test_acc"],
                "delta_vs_nonlip": delta_nonlip,
                "improves_over_nonlip": comparison_label(delta_nonlip),
                "original_best_sigma": None if not best_original else best_original["sigma_init_label"],
                "original_mean": None if not best_original else best_original["mean_test_acc"],
                "delta_vs_original": delta_original,
                "comparable_to_original": comparison_label(delta_original),
            }
        )
    return rows


def _mean_by_sigma(cell_runs: list[dict[str, Any]]) -> dict[Any, float]:
    """Map sigma_init_label -> mean(test_acc) across seeds for a single cell."""
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for run in cell_runs:
        groups[run.get("sigma_init_label")].append(run)
    out: dict[Any, float] = {}
    for sigma, group in groups.items():
        accs = [r["last_test_acc"] for r in group if r.get("last_test_acc") is not None]
        if accs:
            out[sigma] = statistics.fmean(accs)
    return out


def _seed_std_for_cell(cell_runs: list[dict[str, Any]]) -> float | None:
    """Within-sigma seed-std averaged across sigma values. If each sigma bucket
    has >=2 seeds, compute std-across-seeds per sigma then mean; else return
    None. This is the noise floor for comparing against std-across-sigma."""
    by_sigma: dict[Any, list[float]] = defaultdict(list)
    for run in cell_runs:
        acc = run.get("last_test_acc")
        if acc is not None:
            by_sigma[run.get("sigma_init_label")].append(float(acc))
    per_sigma_stds = [
        statistics.pstdev(accs) for accs in by_sigma.values() if len(accs) >= 2
    ]
    if not per_sigma_stds:
        return None
    return statistics.fmean(per_sigma_stds)


def build_sigma_robustness_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each (dataset, partition) and method, compute best-sigma vs avg-sigma
    accuracy, the cross-sigma spread/std, and the cost of using average-sigma
    instead of the best-sigma. Methods: convex-bal ep=1/ep=3 from -4-22,
    lipschitz textbook_sc from paper-lipschitz-estimator. Baseline for the
    "delta vs original" column is sgd_original ep=1 from -4-22 (best sigma)."""
    methods: list[tuple[str, str, list[dict[str, Any]]]] = [
        (
            "sgd_convex_bal_ep1",
            "SGD convex-bal ep=1",
            [r for r in runs if is_sgd_convex_bal_run(r) and int(r.get("epochs") or 0) == 1],
        ),
        (
            "sgd_convex_bal_ep3",
            "SGD convex-bal ep=3",
            [r for r in runs if is_sgd_convex_bal_run(r) and int(r.get("epochs") or 0) == 3],
        ),
        (
            "lipschitz_textbook_sc",
            "Lipschitz textbook_sc",
            [r for r in runs if is_lipschitz_textbook_sc_run(r)],
        ),
    ]
    baseline_runs = scored_runs(run for run in runs if is_sisa_baseline_run(run))

    cells: set[tuple[Any, Any]] = set()
    for _, _, cohort in methods:
        for run in cohort:
            if run.get("last_test_acc") is None:
                continue
            cells.add((run.get("dataset"), run.get("partition")))

    rows: list[dict[str, Any]] = []
    sigma_display = ("1e2", "1e3", "1e4")
    for dataset, partition in sorted(cells):
        baseline_sigma_means = _mean_by_sigma(
            [r for r in baseline_runs if r.get("dataset") == dataset and r.get("partition") == partition]
        )
        baseline_best = max(baseline_sigma_means.values()) if baseline_sigma_means else None
        baseline_best_sigma = (
            max(baseline_sigma_means.items(), key=lambda kv: kv[1])[0]
            if baseline_sigma_means
            else None
        )

        # Collect per-method stats first, then mark Pareto-optimal within this cell.
        cell_method_rows: list[dict[str, Any]] = []
        for method_id, method_label, cohort in methods:
            cell_runs = scored_runs(
                r for r in cohort if r.get("dataset") == dataset and r.get("partition") == partition
            )
            sigma_means = _mean_by_sigma(cell_runs)
            if not sigma_means:
                continue
            vals = list(sigma_means.values())
            best_acc = max(vals)
            worst_acc = min(vals)
            best_sigma = max(sigma_means.items(), key=lambda kv: kv[1])[0]
            worst_sigma = min(sigma_means.items(), key=lambda kv: kv[1])[0]
            avg_acc = statistics.fmean(vals)
            spread = best_acc - worst_acc if len(vals) > 1 else 0.0
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            seed_std = _seed_std_for_cell(cell_runs)
            sigma_vs_seed_ratio = None if seed_std in (None, 0) else std / seed_std
            # Online-learning-style metrics:
            # - tuning_benefit = best - worst. How much does picking the right
            #   sigma buy you vs the worst possible choice? (worst-case
            #   parameter-sensitivity; small = method is anytime-robust).
            # - parameter_free_gap = avg - baseline_best. Does the method beat
            #   a fully-tuned baseline when run untuned (averaged over sigmas)?
            tuning_benefit = best_acc - worst_acc
            cell_method_rows.append(
                {
                    "dataset": dataset,
                    "partition": compact_partition(partition),
                    "method": method_label,
                    "n_sigma": len(sigma_means),
                    "best_sigma": best_sigma,
                    "worst_sigma": worst_sigma,
                    "acc_sigma_1e2": sigma_means.get("1e2"),
                    "acc_sigma_1e3": sigma_means.get("1e3"),
                    "acc_sigma_1e4": sigma_means.get("1e4"),
                    "best_sigma_mean": best_acc,
                    "worst_sigma_mean": worst_acc,
                    "avg_across_sigma": avg_acc,
                    "std_across_sigma": std,
                    "spread_across_sigma": spread,
                    "seed_std": seed_std,
                    "sigma_vs_seed_ratio": sigma_vs_seed_ratio,
                    "tuning_benefit": tuning_benefit,
                    "drop_best_to_avg": best_acc - avg_acc,
                    "baseline_best_sigma": baseline_best_sigma,
                    "baseline_best_mean": baseline_best,
                    "parameter_free_gap": None if baseline_best is None else avg_acc - baseline_best,
                    "worst_vs_baseline_best": None if baseline_best is None else worst_acc - baseline_best,
                    "best_vs_baseline_best": None if baseline_best is None else best_acc - baseline_best,
                    "pareto_optimal": None,
                }
            )

        # Pareto within this cell on (best_sigma_mean, worst_sigma_mean), both
        # maximized. This is the online-learning framing: peak accuracy AND
        # worst-case accuracy must both be non-dominated. A uniformly-bad
        # method is not Pareto-optimal here (its worst AND best are both low),
        # fixing the pathology of the (best, drop) formulation.
        for row in cell_method_rows:
            dominated = False
            for other in cell_method_rows:
                if other is row:
                    continue
                if (
                    other["best_sigma_mean"] >= row["best_sigma_mean"]
                    and other["worst_sigma_mean"] >= row["worst_sigma_mean"]
                    and (
                        other["best_sigma_mean"] > row["best_sigma_mean"]
                        or other["worst_sigma_mean"] > row["worst_sigma_mean"]
                    )
                ):
                    dominated = True
                    break
            row["pareto_optimal"] = "no" if dominated else "yes"

        rows.extend(cell_method_rows)

    rows.sort(key=lambda row: (str(row["dataset"]), str(row["partition"]), row["method"]))
    return rows


def build_partition_robustness_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate section-6 stats across datasets within each partition.
    Reveals whether robustness degrades as heterogeneity increases
    (label3 -> label1)."""
    base_rows = build_sigma_robustness_rows(runs)
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        grouped[(row["partition"], row["method"])].append(row)

    def avg(group: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in group if r.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    partition_order = {"label3": 0, "label2": 1, "label1": 2}
    out: list[dict[str, Any]] = []
    for (partition, method), group in grouped.items():
        out.append(
            {
                "partition": partition,
                "method": method,
                "n_cells": len(group),
                "mean_best_acc": avg(group, "best_sigma_mean"),
                "mean_worst_acc": avg(group, "worst_sigma_mean"),
                "mean_avg_acc": avg(group, "avg_across_sigma"),
                "mean_tuning_benefit": avg(group, "tuning_benefit"),
                "mean_sigma_std": avg(group, "std_across_sigma"),
                "mean_seed_std": avg(group, "seed_std"),
                "mean_parameter_free_gap": avg(group, "parameter_free_gap"),
                "mean_worst_vs_orig": avg(group, "worst_vs_baseline_best"),
                "mean_best_vs_orig": avg(group, "best_vs_baseline_best"),
            }
        )
    out.sort(key=lambda r: (partition_order.get(str(r["partition"]), 9), r["method"]))
    return out


def build_robustness_pareto_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flat (method, cell) list sorted so Pareto-optimal rows come first,
    then by descending best-sigma accuracy. Lets reader scan which methods
    dominate their respective cells."""
    base_rows = build_sigma_robustness_rows(runs)
    out: list[dict[str, Any]] = []
    for row in base_rows:
        out.append(
            {
                "dataset": row["dataset"],
                "partition": row["partition"],
                "method": row["method"],
                "best_sigma": row["best_sigma"],
                "worst_sigma": row["worst_sigma"],
                "best_sigma_mean": row["best_sigma_mean"],
                "worst_sigma_mean": row["worst_sigma_mean"],
                "avg_across_sigma": row["avg_across_sigma"],
                "tuning_benefit": row["tuning_benefit"],
                "seed_std": row["seed_std"],
                "parameter_free_gap": row.get("parameter_free_gap"),
                "worst_vs_baseline_best": row.get("worst_vs_baseline_best"),
                "pareto_optimal": row["pareto_optimal"],
            }
        )
    # Pareto-first; tiebreak on worst-sigma (the conservative floor), then best-sigma.
    out.sort(
        key=lambda r: (
            0 if r["pareto_optimal"] == "yes" else 1,
            -(r["worst_sigma_mean"] or 0.0),
            -(r["best_sigma_mean"] or 0.0),
        )
    )
    return out


def build_warmstart_vs_reset_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair finished cells from `sisa-exact-admm-sgd-epochs-4-22` (reset to
    w_global each round) with the same cells from `sisa-exact-admm-warmstart`
    (warm-start from w_i^{k-1}). Diff = warmstart - reset; negative means
    the warm-start change made the cell worse."""
    by_proj_cell: dict[str, dict[tuple[Any, ...], list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run.get("status") != "finished" or run.get("last_test_acc") is None:
            continue
        proj = run.get("project")
        if proj not in (SGD_EXACT_ADMM_PROJECT, WARMSTART_PROJECT):
            continue
        cfg = run.get("config") or {}
        if cfg.get("optimizer") != "sgd":
            continue
        if cfg.get("sigma_mode") != "online_convex_bal":
            continue
        key = (
            run.get("dataset"),
            run.get("partition"),
            str(cfg.get("epochs")),
            run.get("sigma_init_label"),
        )
        by_proj_cell[proj][key].append(float(run["last_test_acc"]))

    common = set(by_proj_cell[SGD_EXACT_ADMM_PROJECT].keys()) & set(by_proj_cell[WARMSTART_PROJECT].keys())
    rows: list[dict[str, Any]] = []
    for key in sorted(common, key=lambda k: (str(k[0]), str(k[1]), int(k[2] or 0), str(k[3]))):
        ds, part, ep, sig = key
        old = by_proj_cell[SGD_EXACT_ADMM_PROJECT][key]
        new = by_proj_cell[WARMSTART_PROJECT][key]
        old_mean = statistics.fmean(old)
        new_mean = statistics.fmean(new)
        diff = new_mean - old_mean
        if diff <= -0.05:
            verdict = "worse"
        elif diff >= 0.05:
            verdict = "better"
        else:
            verdict = "tied"
        rows.append(
            {
                "dataset": ds,
                "partition": compact_partition(part),
                "epochs": ep,
                "sigma_init_label": sig,
                "n_reset": len(old),
                "n_warmstart": len(new),
                "mean_reset": old_mean,
                "mean_warmstart": new_mean,
                "diff": diff,
                "verdict": verdict,
            }
        )
    rows.sort(key=lambda r: r["diff"])  # worst regressions first
    return rows


def build_variant_vs_sgd_ep1_rows(runs: list[dict[str, Any]], variant_name: str) -> list[dict[str, Any]]:
    if variant_name == "adam_warmstart":
        variant_runs = scored_runs(run for run in runs if is_exact_admm_adam_warmstart_run(run))
    elif variant_name == "task_aware":
        variant_runs = scored_runs(run for run in runs if is_exact_admm_task_aware_run(run))
    else:
        return []

    baseline_runs = scored_runs(
        run for run in runs if is_sgd_convex_bal_run(run) and int(run.get("epochs") or 0) == 1
    )
    baseline_by_cell: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in baseline_runs:
        baseline_by_cell[(run.get("dataset"), run.get("partition"))].append(run)

    variant_by_cell: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in variant_runs:
        variant_by_cell[(run.get("dataset"), run.get("partition"))].append(run)

    rows: list[dict[str, Any]] = []
    for (dataset, partition), cell_runs in sorted(variant_by_cell.items()):
        variant_groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for run in cell_runs:
            variant_groups[(run.get("sigma_init_label"), config_value(run, "task_lambda"))].append(run)
        variant_candidates: list[dict[str, Any]] = []
        for (sigma_label, task_lambda), group_runs in variant_groups.items():
            summary = summarize_scored_runs(group_runs)
            if summary is not None:
                variant_candidates.append(
                    {
                        "sigma_init_label": sigma_label,
                        "task_lambda": task_lambda,
                        **summary,
                    }
                )
        best_variant = choose_best_summary(variant_candidates)

        baseline_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for run in baseline_by_cell.get((dataset, partition), []):
            baseline_groups[run.get("sigma_init_label")].append(run)
        baseline_candidates: list[dict[str, Any]] = []
        for sigma_label, group_runs in baseline_groups.items():
            summary = summarize_scored_runs(group_runs)
            if summary is not None:
                baseline_candidates.append({"sigma_init_label": sigma_label, **summary})
        best_baseline = choose_best_summary(baseline_candidates)

        delta = None if not best_variant or not best_baseline else best_variant["mean_test_acc"] - best_baseline["mean_test_acc"]
        rows.append(
            {
                "dataset": dataset,
                "partition": compact_partition(partition),
                "variant": variant_name.replace("_", " "),
                "best_sigma": None if not best_variant else best_variant["sigma_init_label"],
                "best_task_lambda": None if not best_variant else best_variant.get("task_lambda"),
                "variant_mean": None if not best_variant else best_variant["mean_test_acc"],
                "baseline_sigma": None if not best_baseline else best_baseline["sigma_init_label"],
                "baseline_mean": None if not best_baseline else best_baseline["mean_test_acc"],
                "delta_vs_sgd_ep1": delta,
                "comparison": comparison_label(delta),
            }
        )
    return rows


def sgd_method_label(run: dict[str, Any]) -> str:
    if is_sgd_original_run(run):
        return "original/fixed"
    if is_sgd_convex_bal_run(run):
        return "convex-bal/adaptive"
    return str(run.get("sigma_mode") or "other")


def build_sgd_metric_diagnostic_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sgd_runs = [run for run in runs if run.get("project") == SGD_EXACT_ADMM_PROJECT]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in sgd_runs:
        by_method[sgd_method_label(run)].append(run)

    rows: list[dict[str, Any]] = []
    for method, method_runs in sorted(by_method.items()):
        scored = [run for run in method_runs if run.get("last_test_acc") is not None]
        no_metric = [run for run in method_runs if run.get("last_test_acc") is None]
        sample_no_metric = no_metric[0] if no_metric else None
        sources = Counter(str(run.get("wandb_metric_source") or "none") for run in method_runs)
        statuses = Counter(str(run.get("status") or "unknown") for run in method_runs)
        summary_keys = []
        for run in no_metric:
            summary_keys = run.get("wandb_summary_metric_keys") or []
            if summary_keys:
                break
        rows.append(
            {
                "method": method,
                "total_runs": len(method_runs),
                "usable_acc_runs": len(scored),
                "missing_acc_runs": len(no_metric),
                "finished_missing_acc": sum(1 for run in no_metric if str(run.get("wandb_state") or run.get("status")) == "finished"),
                "status_breakdown": ", ".join(f"{status}:{count}" for status, count in statuses.most_common()),
                "metric_sources": ", ".join(f"{source}:{count}" for source, count in sources.most_common()),
                "sample_summary_keys": ", ".join(summary_keys) if summary_keys else "none found",
                "sample_run": None if sample_no_metric is None else sample_no_metric.get("run_name"),
            }
        )
    return rows


def build_targeted_analysis(runs: list[dict[str, Any]]) -> dict[str, Any]:
    available_projects = sorted({str(run.get("project")) for run in runs if run.get("project")})
    notes = [
        "Lipschitz runs are direct SISA runs that only change the online loss update, while exact-ADMM runs change the local ADMM solver. Cross-family comparisons should be read as context, not as a perfect apples-to-apples ablation.",
    ]
    sgd_project_runs = [run for run in runs if run.get("project") == SGD_EXACT_ADMM_PROJECT]
    if sgd_project_runs:
        sgd_cells = sorted({(run.get("dataset"), run.get("partition")) for run in sgd_project_runs})
        sgd_scored = sum(1 for run in sgd_project_runs if run.get("last_test_acc") is not None)
        notes.append(
            f"{SGD_EXACT_ADMM_PROJECT}: loaded {len(sgd_project_runs)} runs from W&B/local data, "
            f"{sgd_scored} with usable test accuracy, covering cells "
            f"{', '.join(f'{dataset}/{partition}' for dataset, partition in sgd_cells)}."
        )
        sgd_convex_runs = [run for run in sgd_project_runs if is_sgd_convex_bal_run(run)]
        sgd_convex_scored = sum(1 for run in sgd_convex_runs if run.get("last_test_acc") is not None)
        if sgd_convex_runs and sgd_convex_scored == 0:
            notes.append(
                "The SGD convex-bal/adaptive runs are present in W&B config, but none currently expose a readable "
                "test accuracy under the known metric keys. That is why sections 1-3 are empty while the Lipschitz "
                "sections work."
            )
    lipschitz_project_runs = [run for run in runs if run.get("project") == LIPSCHITZ_MAIN_PROJECT]
    if lipschitz_project_runs:
        lip_cells = sorted({(run.get("dataset"), run.get("partition")) for run in lipschitz_project_runs})
        notes.append(
            f"{LIPSCHITZ_MAIN_PROJECT}: loaded {len(lipschitz_project_runs)} runs covering "
            f"{', '.join(f'{dataset}/{partition}' for dataset, partition in lip_cells)}."
        )
    expected_projects = [
        SGD_EXACT_ADMM_PROJECT,
        LIPSCHITZ_MAIN_PROJECT,
        EXACT_ADMM_PROJECT,
    ]
    missing_projects = [project for project in expected_projects if project not in available_projects]
    if missing_projects:
        notes.append(
            "Missing from the current dataset: " + ", ".join(missing_projects) + ". Those sections will stay empty until local logs or wandb fetches provide them."
        )

    sections = [
        {
            "id": "sgd_metric_diagnostics",
            "title": "0. W&B Metric Diagnostics for SGD Sweep",
            "description": "This shows whether the SGD project has run configs but missing accuracy history. If convex-bal has total runs but zero usable accuracy, the comparison tables below cannot be computed yet.",
            "empty_message": f"No diagnostic rows available from {SGD_EXACT_ADMM_PROJECT}.",
            "columns": [
                {"key": "method", "label": "Method"},
                {"key": "total_runs", "label": "Total"},
                {"key": "usable_acc_runs", "label": "Usable Acc"},
                {"key": "missing_acc_runs", "label": "Missing Acc"},
                {"key": "finished_missing_acc", "label": "Finished Missing"},
                {"key": "status_breakdown", "label": "Statuses"},
                {"key": "metric_sources", "label": "Metric Sources"},
                {"key": "sample_summary_keys", "label": "Sample Summary Keys"},
                {"key": "sample_run", "label": "Sample Run"},
            ],
            "rows": build_sgd_metric_diagnostic_rows(runs),
        },
        {
            "id": "sgd_epoch_comparison",
            "title": "1. Convex Bal vs Original By Dataset / Partition / Epoch",
            "description": "For the SGD exact-ADMM sweep, each epoch row uses the best convex-bal sigma0 available in that cell and compares it against the best original-method sigma0 in the same dataset/partition.",
            "empty_message": f"No runs available from {SGD_EXACT_ADMM_PROJECT}.",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "epochs", "label": "Epochs"},
                {"key": "best_sigma", "label": "Best Convex Sigma"},
                {"key": "convex_mean", "label": "Convex Mean", "kind": "metric"},
                {"key": "original_best_sigma", "label": "Best Orig Sigma"},
                {"key": "original_mean", "label": "Orig Mean", "kind": "metric"},
                {"key": "delta_vs_original", "label": "Delta", "kind": "delta"},
                {"key": "comparison", "label": "Comparable?"},
            ],
            "rows": build_sgd_epoch_comparison_rows(runs),
        },
        {
            "id": "sgd_sigma_comparison",
            "title": "2. Convex Bal Sigma0 Comparison vs Original",
            "description": "Within each dataset/partition and sigma0, this table takes the best epoch for that sigma0 and compares it against the best original-method sigma0 for the same cell. This makes the sigma0=1e3 story explicit while still showing 1e2 and 1e4.",
            "empty_message": f"No sigma-comparison rows available from {SGD_EXACT_ADMM_PROJECT}.",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "sigma_init_label", "label": "Sigma0"},
                {"key": "best_epoch", "label": "Best Epoch"},
                {"key": "convex_mean", "label": "Convex Mean", "kind": "metric"},
                {"key": "original_best_sigma", "label": "Best Orig Sigma"},
                {"key": "original_mean", "label": "Orig Mean", "kind": "metric"},
                {"key": "delta_vs_original", "label": "Delta", "kind": "delta"},
                {"key": "comparison", "label": "Comparable?"},
            ],
            "rows": build_sgd_sigma_comparison_rows(runs),
        },
        {
            "id": "sgd_epoch_effect",
            "title": "3. Do More Epochs Help for Convex Bal?",
            "description": "Each epoch value uses its own best sigma0 inside the convex-bal SGD exact-ADMM sweep, then compares the best epoch against epoch 1 for the same dataset/partition.",
            "empty_message": f"No epoch-effect rows available from {SGD_EXACT_ADMM_PROJECT}.",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "ep1_sigma", "label": "Ep1 Sigma"},
                {"key": "ep1_mean", "label": "Ep1 Mean", "kind": "metric"},
                {"key": "ep3_sigma", "label": "Ep3 Sigma"},
                {"key": "ep3_mean", "label": "Ep3 Mean", "kind": "metric"},
                {"key": "ep10_sigma", "label": "Ep10 Sigma"},
                {"key": "ep10_mean", "label": "Ep10 Mean", "kind": "metric"},
                {"key": "best_epoch", "label": "Best Epoch"},
                {"key": "delta_vs_ep1", "label": "Delta vs Ep1", "kind": "delta"},
                {"key": "more_epochs_help", "label": "More Epochs Help?"},
            ],
            "rows": build_sgd_epoch_effect_rows(runs),
        },
        {
            "id": "lipschitz_vs_nonlip",
            "title": "4. Lipschitz Convex Bal vs Non-Lipschitz Convex Bal",
            "description": (
                "Lipschitz convex-bal (paper-lipschitz-estimator) vs non-Lipschitz convex-bal "
                "from the exact-ADMM SGD sweep at epochs=1 (" + SGD_EXACT_ADMM_PROJECT + "). "
                "Both columns report the best sigma0 within their respective method for each "
                "dataset/partition."
            ),
            "empty_message": f"No rows available from {LIPSCHITZ_MAIN_PROJECT} or the exact-ADMM SGD ep=1 convex-bal baseline.",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "lipschitz_best_sigma", "label": "Lip Sigma"},
                {"key": "lipschitz_mean", "label": "Lip Mean", "kind": "metric"},
                {"key": "nonlip_best_sigma", "label": "Non-Lip Sigma"},
                {"key": "nonlip_mean", "label": "Non-Lip Mean", "kind": "metric"},
                {"key": "delta_vs_nonlip", "label": "Delta", "kind": "delta"},
                {"key": "improves_over_nonlip", "label": "Improves?"},
            ],
            "rows": build_lipschitz_comparison_rows(runs),
        },
        {
            "id": "lipschitz_vs_original",
            "title": "5. Lipschitz Convex Bal vs Original SISA",
            "description": (
                "Lipschitz convex-bal (paper-lipschitz-estimator) vs the original SISA baseline "
                "runs (sgd_original) from " + SGD_EXACT_ADMM_PROJECT + ". Each side uses its own "
                "best sigma0 per dataset/partition."
            ),
            "empty_message": f"No original-comparison rows available from {LIPSCHITZ_MAIN_PROJECT} or {SGD_EXACT_ADMM_PROJECT} sgd_original.",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "lipschitz_best_sigma", "label": "Lip Sigma"},
                {"key": "lipschitz_mean", "label": "Lip Mean", "kind": "metric"},
                {"key": "original_best_sigma", "label": "Orig Sigma"},
                {"key": "original_mean", "label": "Orig Mean", "kind": "metric"},
                {"key": "delta_vs_original", "label": "Delta", "kind": "delta"},
                {"key": "comparable_to_original", "label": "Comparable?"},
            ],
            "rows": build_lipschitz_comparison_rows(runs),
        },
        {
            "id": "sigma_robustness_tradeoff",
            "title": "6. σ-Init Robustness vs Accuracy (Online-Learning Framing)",
            "description": (
                "Online-learning-aligned metrics for each dataset/partition/method:"
                "\n • `Worst σ Mean` — anytime/worst-case accuracy across σ∈{1e2,1e3,1e4}. "
                "The practical lower bound without tuning."
                "\n • `Best σ Mean` — peak accuracy if you tune σ."
                "\n • `Tuning Benefit` = Best − Worst. How much picking the right σ "
                "buys you over the worst choice. Small = σ-invariant."
                "\n • `Parameter-free Gap` = Avg-σ − Orig Best. Positive = the untuned "
                "adaptive method beats a fully-tuned non-adaptive baseline (the "
                "central online-learning claim)."
                "\n • `σ/Seed Ratio` = σ-Std / Seed-Std. Below 1 = σ-variation is "
                "indistinguishable from seed noise."
                "\n • `Pareto?` = not dominated on (Best σ Mean, Worst σ Mean) within "
                "the cell. Both axes maximized — uniformly-bad methods are not "
                "rewarded."
                "\nMethods: SGD convex-bal ep=1 and ep=3 from " + SGD_EXACT_ADMM_PROJECT
                + ", Lipschitz textbook_sc from " + LIPSCHITZ_MAIN_PROJECT + "."
            ),
            "empty_message": "No robustness rows (need convex-bal ep=1/ep=3 or lipschitz textbook_sc runs).",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "method", "label": "Method"},
                {"key": "n_sigma", "label": "# σ"},
                {"key": "acc_sigma_1e2", "label": "Acc @ σ=1e2", "kind": "metric"},
                {"key": "acc_sigma_1e3", "label": "Acc @ σ=1e3", "kind": "metric"},
                {"key": "acc_sigma_1e4", "label": "Acc @ σ=1e4", "kind": "metric"},
                {"key": "worst_sigma", "label": "Worst σ"},
                {"key": "worst_sigma_mean", "label": "Worst σ Mean", "kind": "metric"},
                {"key": "best_sigma", "label": "Best σ"},
                {"key": "best_sigma_mean", "label": "Best σ Mean", "kind": "metric"},
                {"key": "avg_across_sigma", "label": "Avg-σ Mean", "kind": "metric"},
                {"key": "tuning_benefit", "label": "Tuning Benefit", "kind": "delta"},
                {"key": "seed_std", "label": "Seed-Std", "kind": "metric"},
                {"key": "sigma_vs_seed_ratio", "label": "σ/Seed Ratio", "kind": "metric"},
                {"key": "baseline_best_mean", "label": "Orig Best", "kind": "metric"},
                {"key": "parameter_free_gap", "label": "Param-Free Gap", "kind": "delta"},
                {"key": "worst_vs_baseline_best", "label": "Δ Worst vs Orig", "kind": "delta"},
                {"key": "best_vs_baseline_best", "label": "Δ Best vs Orig", "kind": "delta"},
                {"key": "pareto_optimal", "label": "Pareto?"},
            ],
            "rows": build_sigma_robustness_rows(runs),
        },
        {
            "id": "partition_heterogeneity",
            "title": "7. Partition Heterogeneity: Does Robustness Scale with Non-IID-ness?",
            "description": (
                "Section 6 aggregated by partition across datasets. Does worst-σ "
                "accuracy hold up as heterogeneity increases (label3 → label1)? "
                "A method that retains worst-σ performance on label1 while others "
                "collapse to noise is the headline robustness claim. `Mean Seed-Std` "
                "is the within-σ seed noise floor — compare against `Mean σ-Std` "
                "to see whether cross-σ variance is real structure or just noise."
            ),
            "empty_message": "No partition-level robustness rows available.",
            "columns": [
                {"key": "partition", "label": "Partition"},
                {"key": "method", "label": "Method"},
                {"key": "n_cells", "label": "# Cells"},
                {"key": "mean_worst_acc", "label": "Mean Worst σ Acc", "kind": "metric"},
                {"key": "mean_best_acc", "label": "Mean Best σ Acc", "kind": "metric"},
                {"key": "mean_avg_acc", "label": "Mean Avg-σ Acc", "kind": "metric"},
                {"key": "mean_tuning_benefit", "label": "Mean Tuning Benefit", "kind": "delta"},
                {"key": "mean_sigma_std", "label": "Mean σ-Std", "kind": "metric"},
                {"key": "mean_seed_std", "label": "Mean Seed-Std", "kind": "metric"},
                {"key": "mean_parameter_free_gap", "label": "Mean Param-Free Gap", "kind": "delta"},
                {"key": "mean_worst_vs_orig", "label": "Mean Δ Worst vs Orig", "kind": "delta"},
                {"key": "mean_best_vs_orig", "label": "Mean Δ Best vs Orig", "kind": "delta"},
            ],
            "rows": build_partition_robustness_rows(runs),
        },
        {
            "id": "robustness_pareto",
            "title": "8. Pareto Ranking: Best σ vs Worst σ Accuracy per Cell",
            "description": (
                "All (method, dataset, partition) entries flattened. Pareto dominance "
                "uses (Best σ Mean, Worst σ Mean), both maximized — the online-learning "
                "framing. `Pareto?=yes` means no other method in the same cell has both "
                "peak AND worst-case accuracy ≥ this one. A uniformly-bad method cannot "
                "be Pareto-optimal here because its worst-σ is also low. Rows sorted "
                "Pareto-first, then by descending worst-σ mean. Methods that are "
                "strictly dominated can be dropped for the corresponding cell."
            ),
            "empty_message": "No Pareto rows available.",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "method", "label": "Method"},
                {"key": "worst_sigma", "label": "Worst σ"},
                {"key": "worst_sigma_mean", "label": "Worst σ Mean", "kind": "metric"},
                {"key": "best_sigma", "label": "Best σ"},
                {"key": "best_sigma_mean", "label": "Best σ Mean", "kind": "metric"},
                {"key": "avg_across_sigma", "label": "Avg-σ Mean", "kind": "metric"},
                {"key": "tuning_benefit", "label": "Tuning Benefit", "kind": "delta"},
                {"key": "seed_std", "label": "Seed-Std", "kind": "metric"},
                {"key": "parameter_free_gap", "label": "Param-Free Gap", "kind": "delta"},
                {"key": "worst_vs_baseline_best", "label": "Δ Worst vs Orig", "kind": "delta"},
                {"key": "pareto_optimal", "label": "Pareto?"},
            ],
            "rows": build_robustness_pareto_rows(runs),
        },
        {
            "id": "warmstart_vs_reset",
            "title": "9. Local-Solve Warm-Start vs Reset (2026-04-24 method change)",
            "description": (
                "<b>Comparison:</b> exact-ADMM SGD with reset-to-w<sub>global</sub> "
                "(<code>" + SGD_EXACT_ADMM_PROJECT + "</code>) vs warm-start-from-w<sub>i</sub> "
                "(<code>" + WARMSTART_PROJECT + "</code>). "
                "<b>Diff = warmstart_mean − reset_mean</b>; negative = warm-start regressed.<br><br>"
                "<b>Headline finding: warm-start regresses ~70%+ of paired cells.</b> "
                "<b>Catastrophic on label1 with ep≥1 across mnist/fmnist:</b> e.g., mnist_label1 "
                "ep=1 dropped 33–43pp, ep=10 dropped 22–28pp; fmnist_label1 ep=10 dropped "
                "14–29pp. <b>Help is narrow — limited to ep=10 on mnist/fmnist label3 "
                "(+5–9pp)</b> and mnist_label2 ep=10 σ∈{1e3,1e4} (+11pp). "
                "<b>cifar10 universally regresses by 2–5pp at all ep, σ</b> — warm-start did "
                "NOT close the gap to tuned SISA. <b>Recommendation: revert the warm-start "
                "change for the main paper method.</b> Lipschitz textbook_sc results in "
                "Sections 6–8 are unaffected by this change (they live in the Lipschitz "
                "project, not the SGD-exact-ADMM project)."
            ),
            "empty_message": (
                f"No comparable runs (need both {SGD_EXACT_ADMM_PROJECT} and {WARMSTART_PROJECT})."
            ),
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "epochs", "label": "Ep"},
                {"key": "sigma_init_label", "label": "σ"},
                {"key": "n_reset", "label": "n (reset)"},
                {"key": "n_warmstart", "label": "n (warm)"},
                {"key": "mean_reset", "label": "Reset Mean", "kind": "metric"},
                {"key": "mean_warmstart", "label": "Warmstart Mean", "kind": "metric"},
                {"key": "diff", "label": "Diff (warm − reset)", "kind": "delta"},
                {"key": "verdict", "label": "Verdict"},
            ],
            "rows": build_warmstart_vs_reset_rows(runs),
        },
        {
            "id": "adam_warmstart_vs_sgd_ep1",
            "title": "Adam Warmstart vs Convex Bal SGD Epoch 1",
            "description": "For exact-ADMM warmstart runs, this compares the best warmstart setting in each dataset/partition against the best convex-bal exact-ADMM SGD run with epochs=1.",
            "empty_message": "No Adam warmstart rows are available in the current dataset.",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "best_sigma", "label": "Adam Sigma"},
                {"key": "variant_mean", "label": "Adam Mean", "kind": "metric"},
                {"key": "baseline_sigma", "label": "SGD Sigma"},
                {"key": "baseline_mean", "label": "SGD Ep1 Mean", "kind": "metric"},
                {"key": "delta_vs_sgd_ep1", "label": "Delta", "kind": "delta"},
                {"key": "comparison", "label": "Improves?"},
            ],
            "rows": build_variant_vs_sgd_ep1_rows(runs, "adam_warmstart"),
        },
        {
            "id": "task_aware_vs_sgd_ep1",
            "title": "Task-Aware Loss vs Convex Bal SGD Epoch 1",
            "description": "For exact-ADMM task-aware runs, this compares the best task-aware setting in each dataset/partition against the best convex-bal exact-ADMM SGD run with epochs=1.",
            "empty_message": "No task-aware rows are available in the current dataset.",
            "columns": [
                {"key": "dataset", "label": "Dataset"},
                {"key": "partition", "label": "Partition"},
                {"key": "best_sigma", "label": "Task Sigma"},
                {"key": "best_task_lambda", "label": "Task Lambda"},
                {"key": "variant_mean", "label": "Task Mean", "kind": "metric"},
                {"key": "baseline_sigma", "label": "SGD Sigma"},
                {"key": "baseline_mean", "label": "SGD Ep1 Mean", "kind": "metric"},
                {"key": "delta_vs_sgd_ep1", "label": "Delta", "kind": "delta"},
                {"key": "comparison", "label": "Improves?"},
            ],
            "rows": build_variant_vs_sgd_ep1_rows(runs, "task_aware"),
        },
    ]
    # Trim to the subset that directly tests the online-learning theorems:
    #   (a) sigma-regret shrinks to noise floor (section 6 columns
    #       std_across_sigma / seed_std / sigma_vs_seed_ratio)
    #   (b) parameter-free claim: untuned avg-sigma accuracy vs tuned baseline
    #       (section 6 column parameter_free_gap)
    #   (c) robustness holds across heterogeneity levels (section 7)
    #   (d) method is Pareto-optimal on (best, worst) vs adaptive competitors
    #       (section 8)
    # Other sections (sgd epoch/sigma comparisons, lipschitz-vs-nonlip,
    # adam/task-aware) are exploratory and don't test the theorems.
    THEOREM_SECTION_IDS = {
        "sigma_robustness_tradeoff",
        "partition_heterogeneity",
        "robustness_pareto",
        "warmstart_vs_reset",
    }
    sections = [s for s in sections if s["id"] in THEOREM_SECTION_IDS]
    return {
        "notes": notes,
        "sections": sections,
        "available_projects": available_projects,
    }


def build_insights(runs: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[str]:
    insights: list[str] = []
    status_counts = Counter(run["status"] for run in runs)
    if status_counts.get("failed"):
        disk_failures = sum(1 for run in runs if run["has_disk_error"])
        insights.append(f"{status_counts['failed']} runs are marked failed; {disk_failures} mention disk pressure explicitly.")
    partial = status_counts.get("partial", 0)
    if partial:
        insights.append(f"{partial} runs are still partial, so top-line tables can shift as logs continue to grow.")

    sparse = [row for row in summaries if row["n_runs"] == 1]
    if sparse:
        insights.append(f"{len(sparse)} summary cells currently have only one run, so their means are single-seed placeholders.")

    by_dataset_partition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_dataset_partition[(row["dataset"], row["partition"])].append(row)
    for (dataset, partition), rows in sorted(by_dataset_partition.items()):
        top = max(rows, key=lambda row: row["mean_test_acc"])
        insights.append(
            f"Best current mean for {dataset} {partition}: {top['mean_test_acc']:.3f} from {top['project']} "
            f"({top['sigma_mode']}, ep={top['epochs']}, sigma={top['sigma_init_label']})."
        )
        if len(insights) >= 6:
            break
    return insights[:8]


def build_payload(runs: list[dict[str, Any]], source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    summaries = build_group_summaries(runs)
    best_cells = build_best_cells(runs)
    individual_ranking = build_individual_ranking(runs)
    combo_leaders = build_setup_combo_leaders(summaries)
    combined_setup_ranking = build_combined_setup_ranking(summaries)
    targeted_analysis = build_targeted_analysis(runs)
    updated_candidates = [run["updated_at"] for run in runs if run["updated_at"]]
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_dir": str(LOG_DIR),
        "source_metadata": source_metadata or {},
        "overview": {
            "total_runs": len(runs),
            "finished_runs": sum(1 for run in runs if run["status"] == "finished"),
            "partial_runs": sum(1 for run in runs if run["status"] == "partial"),
            "failed_runs": sum(1 for run in runs if run["status"] == "failed"),
            "missing_metric_runs": sum(1 for run in runs if run["last_test_acc"] is None),
            "latest_log_update": max(updated_candidates) if updated_candidates else None,
            "projects": sorted({run["project"] for run in runs if run["project"]}),
            "datasets": sorted({run["dataset"] for run in runs if run["dataset"]}),
            "remote_runs": sum(1 for run in runs if run.get("data_source") in {"wandb", "merged"}),
            "ranked_individual_runs": len(individual_ranking),
            "ranked_setups": len(combined_setup_ranking),
            "excluded_runs": int((source_metadata or {}).get("excluded_runs") or 0),
        },
        "insights": build_insights(runs, summaries),
        "runs": runs,
        "summary_rows": summaries,
        "best_cells": best_cells,
        "individual_ranking": individual_ranking,
        "combo_leaders": combo_leaders,
        "combined_setup_ranking": combined_setup_ranking,
        "targeted_analysis": targeted_analysis,
    }
    return payload


def render_html(payload: dict[str, Any]) -> str:
    embedded = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment Results Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255, 252, 246, 0.92);
      --ink: #182126;
      --muted: #59656d;
      --line: rgba(24, 33, 38, 0.12);
      --accent: #bc4b33;
      --accent-soft: rgba(188, 75, 51, 0.12);
      --good: #1f7a4d;
      --warn: #a46a00;
      --bad: #a0342a;
      --shadow: 0 22px 50px rgba(24, 33, 38, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(188, 75, 51, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(49, 99, 140, 0.14), transparent 24%),
        linear-gradient(180deg, #f9f4ec 0%, var(--bg) 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }}
    .shell {{
      width: min(1380px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .hero {{
      padding: 28px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -80px -80px auto;
      width: 260px;
      height: 260px;
      background: radial-gradient(circle, rgba(188, 75, 51, 0.18), transparent 70%);
      pointer-events: none;
    }}
    h1, h2 {{ margin: 0; }}
    h1 {{
      font-size: clamp(2rem, 3vw, 3.5rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
      max-width: 12ch;
    }}
    .sub {{
      color: var(--muted);
      font-size: 1rem;
      max-width: 78ch;
      margin-top: 14px;
    }}
    .meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      margin-top: 18px;
    }}
    .cards {{
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }}
    .card {{
      padding: 18px;
      border-radius: 18px;
      background: rgba(255,255,255,0.55);
      border: 1px solid var(--line);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .card .value {{
      margin-top: 8px;
      font-size: 1.8rem;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    .panel {{
      padding: 22px;
      margin-top: 18px;
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}
    .panel-head p {{
      margin: 6px 0 0;
      color: var(--muted);
    }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    label {{
      display: block;
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.85);
      color: var(--ink);
      font: inherit;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    tbody tr:hover {{
      background: rgba(188, 75, 51, 0.05);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      border: 1px solid transparent;
      background: var(--accent-soft);
      color: var(--accent);
      white-space: nowrap;
    }}
    .status-finished {{ color: var(--good); background: rgba(31, 122, 77, 0.1); }}
    .status-partial {{ color: var(--warn); background: rgba(164, 106, 0, 0.12); }}
    .status-failed, .status-missing-log, .status-no-metric {{ color: var(--bad); background: rgba(160, 52, 42, 0.1); }}
    .insights {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .insight {{
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.55);
      color: var(--ink);
    }}
    .stack {{
      display: grid;
      gap: 16px;
    }}
    .subsection {{
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.55);
    }}
    .subsection h3 {{
      margin: 0;
      font-size: 1.05rem;
    }}
    .subsection p {{
      margin: 6px 0 12px;
      color: var(--muted);
    }}
    .small {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .mono {{
      font-family: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
      font-size: 0.88rem;
    }}
    @media (max-width: 900px) {{
      .shell {{ width: min(100vw - 18px, 1380px); margin: 10px auto 22px; }}
      .hero, .panel {{ padding: 18px; border-radius: 18px; }}
      table {{ font-size: 0.88rem; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Experiment Results Dashboard</h1>
      <p class="sub">A live dashboard for your PISA data-heterogeneity runs. It can rebuild from paired <span class="mono">experiment_arguments-*.json</span> and <span class="mono">experiment_log-*.log</span>, augment those rows with wandb state, or operate directly from wandb when run in an environment that has API access. In watch mode, the backing JSON is regenerated on an interval so the page stays current.</p>
      <div class="meta" id="hero-meta"></div>
      <div class="grid cards" id="overview-cards"></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>Filters</h2>
          <p>Slice the dashboard the same way you compare cells in the markdown findings.</p>
        </div>
        <div class="small" id="filter-count"></div>
      </div>
      <div class="filters" id="filters"></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>Auto Insights</h2>
          <p>Fast readouts from the current dataset, including merged wandb state when enabled.</p>
        </div>
      </div>
      <div class="insights" id="insights"></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>Theorem Validation</h2>
          <p>Only tables that directly test the online-learning claims: (a) σ-regret shrinks to the seed-noise floor (σ/seed ratio &lt; 1), (b) the untuned avg-σ accuracy competes with the best-σ-tuned baseline (parameter-free gap), (c) robustness holds across heterogeneity levels (label3 → label1), (d) the method is Pareto-optimal against adaptive-σ competitors on (best σ, worst σ).</p>
        </div>
      </div>
      <div class="insights" id="analysis-notes"></div>
      <div class="stack" id="analysis-sections"></div>
    </section>

  </div>

  <script id="initial-data" type="application/json">{embedded}</script>
  <script>
    const dataUrl = "results_dashboard_data.json";
    let state = JSON.parse(document.getElementById("initial-data").textContent);
    const filterKeys = [
      ["project", "Project"],
      ["family", "Family"],
      ["dataset", "Dataset"],
      ["partition", "Partition"],
      ["optimizer", "Optimizer"],
      ["sigma_mode", "Sigma Mode"],
      ["epochs", "Epochs"],
      ["sigma_init_label", "Sigma Init"]
    ];
    const filters = Object.fromEntries(filterKeys.map(([key]) => [key, "all"]));

    const fmt = (value, digits = 3) => value == null ? "n/a" : Number(value).toFixed(digits);
    const sortUnique = values => [...new Set(values.filter(v => v !== null && v !== undefined && v !== ""))].sort((a, b) => String(a).localeCompare(String(b), undefined, {{ numeric: true }}));

    function renderCards() {{
      const overview = state.overview;
      const cards = [
        ["Total runs", overview.total_runs],
        ["Finished", overview.finished_runs],
        ["Partial", overview.partial_runs],
        ["Failed", overview.failed_runs],
        ["Missing metric", overview.missing_metric_runs],
        ["Remote-backed", overview.remote_runs],
        ["Excluded", overview.excluded_runs || 0]
      ];
      document.getElementById("overview-cards").innerHTML = cards.map(([label, value]) => `
        <div class="card">
          <div class="label">${{label}}</div>
          <div class="value">${{value}}</div>
        </div>`).join("");
      const projects = overview.projects.join(", ") || "n/a";
      const source = state.source_metadata || {{}};
      document.getElementById("hero-meta").innerHTML = `
        <span><strong>Generated:</strong> ${{new Date(state.generated_at).toLocaleString()}}</span>
        <span><strong>Latest log update:</strong> ${{overview.latest_log_update ? new Date(overview.latest_log_update).toLocaleString() : "n/a"}}</span>
        <span><strong>Projects:</strong> ${{projects}}</span>
        <span><strong>Source:</strong> ${{source.source_mode || "local"}}${{source.wandb_enabled ? " + wandb" : ""}}</span>`;
    }}

    function renderFilters() {{
      const root = document.getElementById("filters");
      root.innerHTML = filterKeys.map(([key, label]) => {{
        const options = sortUnique(state.runs.map(run => run[key]));
        return `
          <div>
            <label for="filter-${{key}}">${{label}}</label>
            <select id="filter-${{key}}" data-key="${{key}}">
              <option value="all">All</option>
              ${{options.map(option => `<option value="${{option}}">${{option}}</option>`).join("")}}
            </select>
          </div>`;
      }}).join("");
      root.querySelectorAll("select").forEach(select => {{
        select.value = filters[select.dataset.key];
        select.addEventListener("change", () => {{
          filters[select.dataset.key] = select.value;
          renderTables();
        }});
      }});
    }}

    function filteredRuns() {{
      return state.runs.filter(run => filterKeys.every(([key]) => filters[key] === "all" || String(run[key]) === filters[key]));
    }}

    function renderInsights() {{
      document.getElementById("insights").innerHTML = state.insights.map(text => `<div class="insight">${{text}}</div>`).join("");
    }}

    function formatCell(value, kind) {{
      if (value == null || value === "") return "n/a";
      if (kind === "metric") return fmt(value);
      if (kind === "delta") {{
        const number = Number(value);
        const prefix = number > 0 ? "+" : "";
        return `${{prefix}}${{number.toFixed(3)}}`;
      }}
      return String(value);
    }}

    function renderTargetedAnalysis() {{
      const analysis = state.targeted_analysis || {{ notes: [], sections: [] }};
      document.getElementById("analysis-notes").innerHTML = (analysis.notes || [])
        .map(text => `<div class="insight">${{text}}</div>`)
        .join("");

      document.getElementById("analysis-sections").innerHTML = (analysis.sections || []).map(section => {{
        const rows = section.rows || [];
        const tableRows = rows.map(row => `
          <tr>
            ${{section.columns.map(column => `<td>${{formatCell(row[column.key], column.kind)}}</td>`).join("")}}
          </tr>`).join("");
        const body = rows.length
          ? `<table>
              <thead>
                <tr>${{section.columns.map(column => `<th>${{column.label}}</th>`).join("")}}</tr>
              </thead>
              <tbody>${{tableRows}}</tbody>
            </table>`
          : `<p class="small">${{section.empty_message || "No rows available."}}</p>`;
        return `
          <div class="subsection">
            <h3>${{section.title}}</h3>
            <p>${{section.description || ""}}</p>
            ${{body}}
          </div>`;
      }}).join("");
    }}

    function aggregateRuns(runs) {{
      const bucket = new Map();
      for (const run of runs) {{
        if (run.last_test_acc == null) continue;
        const key = [
          run.family,
          run.project,
          run.dataset,
          run.partition,
          run.optimizer,
          run.sigma_mode,
          run.epochs,
          run.sigma_init_label
        ].join("||");
        if (!bucket.has(key)) {{
          bucket.set(key, []);
        }}
        bucket.get(key).push(run);
      }}
      return [...bucket.values()].map(rows => {{
        const scores = rows.map(row => row.last_test_acc);
        return {{
          family: rows[0].family,
          project: rows[0].project,
          dataset: rows[0].dataset,
          partition: rows[0].partition,
          optimizer: rows[0].optimizer,
          sigma_mode: rows[0].sigma_mode,
          epochs: rows[0].epochs,
          sigma_init_label: rows[0].sigma_init_label,
          n_runs: rows.length,
          mean_test_acc: scores.reduce((a, b) => a + b, 0) / scores.length,
          min_test_acc: Math.min(...scores),
          max_test_acc: Math.max(...scores),
          spread: Math.max(...scores) - Math.min(...scores),
          finished_runs: rows.filter(row => row.status === "finished").length,
          partial_runs: rows.filter(row => row.status === "partial").length,
        }};
      }}).sort((a, b) => b.mean_test_acc - a.mean_test_acc);
    }}

    function buildIndividualRanking(runs) {{
      return runs
        .filter(run => run.last_test_acc != null)
        .slice()
        .sort((a, b) =>
          (b.last_test_acc - a.last_test_acc) ||
          String(a.dataset).localeCompare(String(b.dataset)) ||
          String(a.partition).localeCompare(String(b.partition)) ||
          String(a.project).localeCompare(String(b.project)) ||
          String(a.run_name).localeCompare(String(b.run_name))
        )
        .map((run, index) => ({{
          rank: index + 1,
          ...run,
        }}));
    }}

    function buildCombinedSetupRanking(grouped) {{
      const bucket = new Map();
      for (const row of grouped) {{
        const key = [
          row.project,
          row.family,
          row.optimizer,
          row.sigma_mode,
          row.epochs,
          row.sigma_init_label,
        ].join("||");
        if (!bucket.has(key)) {{
          bucket.set(key, []);
        }}
        bucket.get(key).push(row);
      }}

      return [...bucket.values()].map(rows => {{
        const means = rows.map(row => row.mean_test_acc);
        const bestCombo = rows.slice().sort((a, b) => b.mean_test_acc - a.mean_test_acc)[0];
        return {{
          project: rows[0].project,
          family: rows[0].family,
          optimizer: rows[0].optimizer,
          sigma_mode: rows[0].sigma_mode,
          epochs: rows[0].epochs,
          sigma_init_label: rows[0].sigma_init_label,
          combo_count: rows.length,
          datasets_covered: new Set(rows.map(row => row.dataset)).size,
          partitions_covered: new Set(rows.map(row => row.partition)).size,
          overall_mean: means.reduce((a, b) => a + b, 0) / means.length,
          best_combo_mean: bestCombo.mean_test_acc,
          best_combo_dataset: bestCombo.dataset,
          best_combo_partition: bestCombo.partition,
          total_runs: rows.reduce((sum, row) => sum + row.n_runs, 0),
          finished_runs: rows.reduce((sum, row) => sum + row.finished_runs, 0),
          partial_runs: rows.reduce((sum, row) => sum + row.partial_runs, 0),
        }};
      }})
      .sort((a, b) =>
        (b.overall_mean - a.overall_mean) ||
        (b.combo_count - a.combo_count) ||
        (b.best_combo_mean - a.best_combo_mean) ||
        String(a.project).localeCompare(String(b.project))
      )
      .map((row, index) => ({{
        rank: index + 1,
        ...row,
      }}));
    }}

    function buildComboLeaders(grouped) {{
      const bucket = new Map();
      for (const row of grouped) {{
        const key = `${{row.dataset}}||${{row.partition}}`;
        if (!bucket.has(key)) {{
          bucket.set(key, []);
        }}
        bucket.get(key).push(row);
      }}

      return [...bucket.values()].map(rows => {{
        const top = rows.slice().sort((a, b) =>
          (b.mean_test_acc - a.mean_test_acc) ||
          (b.max_test_acc - a.max_test_acc) ||
          (b.n_runs - a.n_runs) ||
          String(a.project).localeCompare(String(b.project))
        )[0];
        return top;
      }})
      .sort((a, b) =>
        (b.mean_test_acc - a.mean_test_acc) ||
        String(a.dataset).localeCompare(String(b.dataset)) ||
        String(a.partition).localeCompare(String(b.partition))
      )
      .map((row, index) => ({{
        rank: index + 1,
        ...row,
      }}));
    }}

    function table(headers, rows) {{
      if (!rows.length) {{
        return `<p class="small">No rows match the current filters.</p>`;
      }}
      return `
        <table>
          <thead>
            <tr>${{headers.map(header => `<th>${{header}}</th>`).join("")}}</tr>
          </thead>
          <tbody>${{rows.join("")}}</tbody>
        </table>`;
    }}

    function renderTables() {{
      // Trimmed dashboard: only the "Theorem Validation" panel is rendered
      // (via renderTargetedAnalysis). The exploratory tables below are kept
      // as dead code so refreshes / filter recomputation don't need to change,
      // but they early-return when their DOM elements don't exist.
      const runs = filteredRuns();
      const filterCountEl = document.getElementById("filter-count");
      if (filterCountEl) filterCountEl.textContent = `${{runs.length}} matching runs`;
      if (!document.getElementById("summary-table")) return;

      const grouped = aggregateRuns(runs);
      const individualRanking = buildIndividualRanking(runs);
      const combinedSetupRanking = buildCombinedSetupRanking(grouped);
      const comboLeaders = buildComboLeaders(grouped);
      const summaryRows = grouped.slice(0, 200).map(row => `
        <tr>
          <td>${{row.project || "n/a"}}</td>
          <td>${{row.dataset || "n/a"}}</td>
          <td>${{row.partition || "n/a"}}</td>
          <td>${{row.optimizer || "n/a"}}</td>
          <td>${{row.sigma_mode || "n/a"}}</td>
          <td>${{row.epochs ?? "n/a"}}</td>
          <td>${{row.sigma_init_label}}</td>
          <td>${{row.n_runs}}</td>
          <td>${{fmt(row.mean_test_acc)}}</td>
          <td>${{fmt(row.min_test_acc)}}</td>
          <td>${{fmt(row.max_test_acc)}}</td>
          <td>${{fmt(row.spread)}}</td>
        </tr>`);
      document.getElementById("summary-table").innerHTML = table(
        ["Project", "Dataset", "Partition", "Opt", "Sigma Mode", "Ep", "Sigma", "n", "Mean", "Min", "Max", "Spread"],
        summaryRows
      );

      const individualRows = individualRanking.slice(0, 100).map(run => `
        <tr>
          <td>${{run.rank}}</td>
          <td>${{fmt(run.last_test_acc)}}</td>
          <td>${{run.dataset || "n/a"}}</td>
          <td>${{run.partition || "n/a"}}</td>
          <td>${{run.project || "n/a"}}</td>
          <td>${{run.optimizer || "n/a"}}</td>
          <td>${{run.sigma_mode || "n/a"}}</td>
          <td>${{run.epochs ?? "n/a"}}</td>
          <td>${{run.sigma_init_label}}</td>
          <td>${{run.seed ?? "n/a"}}</td>
          <td><span class="pill status-${{run.status}}">${{run.status}}</span></td>
          <td class="mono">${{run.run_name || "n/a"}}</td>
        </tr>`);
      document.getElementById("individual-ranking-table").innerHTML = table(
        ["Rank", "Acc", "Dataset", "Partition", "Project", "Opt", "Sigma Mode", "Ep", "Sigma", "Seed", "Status", "Run Name"],
        individualRows
      );

      const combinedRows = combinedSetupRanking.slice(0, 100).map(row => `
        <tr>
          <td>${{row.rank}}</td>
          <td>${{fmt(row.overall_mean)}}</td>
          <td>${{row.combo_count}}</td>
          <td>${{row.project || "n/a"}}</td>
          <td>${{row.optimizer || "n/a"}}</td>
          <td>${{row.sigma_mode || "n/a"}}</td>
          <td>${{row.epochs ?? "n/a"}}</td>
          <td>${{row.sigma_init_label}}</td>
          <td>${{row.best_combo_dataset || "n/a"}}</td>
          <td>${{row.best_combo_partition || "n/a"}}</td>
          <td>${{fmt(row.best_combo_mean)}}</td>
          <td>${{row.total_runs}}</td>
        </tr>`);
      document.getElementById("combined-setup-table").innerHTML = table(
        ["Rank", "Overall Mean", "Combos", "Project", "Opt", "Sigma Mode", "Ep", "Sigma", "Best Dataset", "Best Partition", "Best Cell Mean", "Runs"],
        combinedRows
      );

      const comboLeaderRows = comboLeaders.slice(0, 100).map(row => `
        <tr>
          <td>${{row.rank}}</td>
          <td>${{row.dataset || "n/a"}}</td>
          <td>${{row.partition || "n/a"}}</td>
          <td>${{fmt(row.mean_test_acc)}}</td>
          <td>${{row.project || "n/a"}}</td>
          <td>${{row.optimizer || "n/a"}}</td>
          <td>${{row.sigma_mode || "n/a"}}</td>
          <td>${{row.epochs ?? "n/a"}}</td>
          <td>${{row.sigma_init_label}}</td>
          <td>${{row.n_runs}}</td>
        </tr>`);
      document.getElementById("combo-leaders-table").innerHTML = table(
        ["Rank", "Dataset", "Partition", "Mean", "Project", "Opt", "Sigma Mode", "Ep", "Sigma", "n"],
        comboLeaderRows
      );

      const bestByCell = new Map();
      for (const run of runs) {{
        if (run.last_test_acc == null) continue;
        const key = `${{run.dataset}}||${{run.partition}}`;
        const prev = bestByCell.get(key);
        if (!prev || run.last_test_acc > prev.last_test_acc) {{
          bestByCell.set(key, run);
        }}
      }}
      const bestRows = [...bestByCell.values()]
        .sort((a, b) => String(a.dataset).localeCompare(String(b.dataset)) || String(a.partition).localeCompare(String(b.partition)))
        .map(run => `
          <tr>
            <td>${{run.dataset || "n/a"}}</td>
            <td>${{run.partition || "n/a"}}</td>
            <td>${{fmt(run.last_test_acc)}}</td>
            <td>${{run.project || "n/a"}}</td>
            <td>${{run.sigma_mode || "n/a"}}</td>
            <td>${{run.optimizer || "n/a"}}</td>
            <td>${{run.epochs ?? "n/a"}}</td>
            <td>${{run.sigma_init_label}}</td>
            <td>${{run.seed ?? "n/a"}}</td>
            <td><span class="pill status-${{run.status}}">${{run.status}}</span></td>
          </tr>`);
      document.getElementById("best-table").innerHTML = table(
        ["Dataset", "Partition", "Best Acc", "Project", "Sigma Mode", "Opt", "Ep", "Sigma", "Seed", "Status"],
        bestRows
      );

      const runRows = runs
        .slice()
        .sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)))
        .slice(0, 300)
        .map(run => `
          <tr>
            <td class="mono">${{run.timestamp}}</td>
            <td><span class="pill status-${{run.status}}">${{run.status}}</span></td>
            <td>${{run.project || "n/a"}}</td>
            <td>${{run.dataset || "n/a"}}</td>
            <td>${{run.partition || "n/a"}}</td>
            <td>${{run.optimizer || "n/a"}}</td>
            <td>${{run.sigma_mode || "n/a"}}</td>
            <td>${{run.epochs ?? "n/a"}}</td>
            <td>${{run.sigma_init_label}}</td>
            <td>${{fmt(run.last_test_acc)}}</td>
            <td>${{fmt(run.best_test_acc)}}</td>
            <td>${{run.max_round ?? "n/a"}} / ${{run.comm_round ?? "n/a"}}</td>
            <td>${{run.data_source || "n/a"}}</td>
            <td>${{run.wandb_metric_source || "n/a"}}</td>
            <td class="mono">${{run.run_name || "n/a"}}</td>
          </tr>`);
      document.getElementById("runs-table").innerHTML = table(
        ["Timestamp", "Status", "Project", "Dataset", "Partition", "Opt", "Sigma Mode", "Ep", "Sigma", "Last Acc", "Best Acc", "Round", "Source", "Metric Source", "Run Name"],
        runRows
      );
    }}

    async function refreshData() {{
      try {{
        const response = await fetch(`${{dataUrl}}?t=${{Date.now()}}`, {{ cache: "no-store" }});
        if (!response.ok) return;
        state = await response.json();
        renderCards();
        renderInsights();
        renderTargetedAnalysis();
        renderFilters();
        renderTables();
      }} catch (error) {{
        // File:// loads can't fetch JSON; the embedded payload still renders.
      }}
    }}

    renderCards();
    renderInsights();
    renderTargetedAnalysis();
    renderFilters();
    renderTables();
    setInterval(refreshData, 30000);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local experiment dashboard from argument/log files.")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-html", type=Path, default=OUT_HTML)
    parser.add_argument("--source", choices=("auto", "local", "wandb"), default="auto")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", action="append", dest="wandb_projects", help="Repeat to limit wandb sync to specific projects.")
    parser.add_argument(
        "--project-preset",
        action="append",
        choices=sorted(PROJECT_PRESETS),
        dest="project_presets",
        help="Convenience preset for common project bundles.",
    )
    parser.add_argument("--wandb-timeout", type=int, default=30)
    parser.add_argument(
        "--wandb-full-project-scan",
        action="store_true",
        default=True,
        help="(default) Fetch every run in the selected W&B projects.",
    )
    parser.add_argument(
        "--wandb-filter-by-local-names",
        action="store_true",
        help="Narrow W&B fetches to run names that appear in local logs.",
    )
    parser.add_argument(
        "--wandb-progress",
        action="store_true",
        help="Print per-project W&B fetch progress.",
    )
    parser.add_argument("--watch", action="store_true", help="Continuously rebuild the dashboard files on an interval.")
    parser.add_argument("--refresh-seconds", type=int, default=30, help="Polling interval for --watch mode.")
    args = parser.parse_args()
    selected_projects = expand_project_selection(args.wandb_projects, args.project_presets)

    def run_once() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        source_metadata, runs = build_dashboard(
            out_json=args.out_json,
            out_html=args.out_html,
            source=args.source,
            wandb_entity=args.wandb_entity,
            wandb_projects=selected_projects,
            wandb_timeout=args.wandb_timeout,
            prefer_local_run_names=args.wandb_filter_by_local_names,
            progress=args.wandb_progress,
        )
        print(f"Wrote {args.out_json}")
        print(f"Wrote {args.out_html}")
        print(f"Parsed {len(runs)} runs from {LOG_DIR}")
        print(f"Source mode: {source_metadata.get('source_mode')}")
        if source_metadata.get("wandb_enabled"):
            print(
                f"Wandb sync enabled for {source_metadata.get('wandb_entity')} "
                f"across {len(source_metadata.get('wandb_projects') or [])} projects"
            )
            if source_metadata.get("wandb_projects"):
                print("Projects:", ", ".join(source_metadata["wandb_projects"]))
        elif source_metadata.get("wandb_error"):
            print(f"Wandb sync unavailable: {source_metadata['wandb_error']}")
        return source_metadata, runs

    if not args.watch:
        run_once()
        return

    interval = max(5, int(args.refresh_seconds))
    print(f"Watching dashboard sources every {interval}s. Press Ctrl+C to stop.")
    while True:
        started_at = datetime.now(tz=timezone.utc).isoformat()
        try:
            run_once()
            print(f"Refresh completed at {started_at}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[{started_at}] Refresh failed: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
