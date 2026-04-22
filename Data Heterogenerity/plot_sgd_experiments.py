"""
Plot SGD-epochs experiment metrics from locally saved CSVs.

Each run produced a CSV by experiment_sisa_practise_admm.py or
experiment_sisa_practise_wandb.py when --local_log_dir was set.
This tool scans that directory, filters runs, optionally facets them
into a grid of subplots, and plots a chosen metric.

Filter flags (AND-combined; repeat for OR):
    --dataset, --partition, --method, --sigma-lr, --epochs, --lr, --seed

Grouping flags:
    --facet-by  comma-separated fields; one subplot per distinct combination
                (e.g. --facet-by dataset,partition or --facet-by lr)
    --hue-by    comma-separated fields used to label/color curves within a
                subplot; defaults to all varying fields not in --facet-by
                and not 'seed' (when --agg-seed is on).
    --agg-seed  mean + min/max band across seeds within each hue group.

Examples:
    # One subplot per (dataset, partition); curves colored by lr; mean over seeds
    python plot_sgd_experiments.py --log-dir .../local_metrics \\
        --method sgd_adaptive --epochs 10 \\
        --facet-by dataset,partition --hue-by lr --agg-seed \\
        --metric test_acc --out facet_by_dataset_partition.png

    # One subplot per lr; curves colored by sigma_lr_init
    python plot_sgd_experiments.py --log-dir .../local_metrics \\
        --dataset fmnist --partition noniid-#label1 \\
        --facet-by lr --hue-by sigma_lr_init --agg-seed \\
        --metric test_acc --out fmnist_l1_by_lr.png
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


FIELDS = ["dataset", "partition", "method", "sigma_mode",
          "sigma_lr_init", "epochs", "lr", "seed"]


def load_runs(log_dir):
    """Return list of (meta_dict, rows_list_of_dict) for every CSV in log_dir."""
    import csv

    runs = []
    csv_paths = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    for csv_path in csv_paths:
        meta_path = csv_path[:-len(".csv")] + ".meta.json"
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[warn] failed to read {meta_path}: {e}", file=sys.stderr)

        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        meta["_csv_path"] = csv_path
        meta["_run_tag"] = os.path.basename(csv_path)[:-len(".csv")]
        runs.append((meta, rows))
    return runs


def parse_run_key(rows):
    if not rows:
        return {}
    r0 = rows[0]
    return {k: r0.get(k, "") for k in FIELDS}


def passes_filters(key, filters):
    for field, allowed in filters.items():
        if not allowed:
            continue
        val = str(key.get(field, ""))
        if val not in {str(a) for a in allowed}:
            return False
    return True


def extract_metric(rows, metric):
    xs, ys = [], []
    for r in rows:
        x = r.get("round", "")
        y = r.get(metric, "")
        if x == "" or y == "":
            continue
        try:
            xs.append(int(x))
            ys.append(float(y))
        except ValueError:
            continue
    return np.array(xs), np.array(ys)


def split_fields(spec):
    if not spec:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def fmt_combo(fields, values):
    return ", ".join(f"{f}={v}" for f, v in zip(fields, values)) or "all"


def main():
    parser = argparse.ArgumentParser(
        description="Plot SGD-experiment CSV metrics saved by experiment_sisa_practise_*.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-dir", required=True,
                        help="Directory with per-run CSVs (and .meta.json).")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--partition", action="append", default=[])
    parser.add_argument("--method", action="append", default=[],
                        help="Matches 'method' or 'sigma_mode' column.")
    parser.add_argument("--sigma-lr", action="append", default=[])
    parser.add_argument("--epochs", action="append", default=[])
    parser.add_argument("--lr", action="append", default=[])
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--metric", default="test_acc",
                        choices=["test_acc", "train_local_admm_loss_avg",
                                 "primal_res_avg", "delta_w_global_avg",
                                 "sigma_value", "log_sigma_value",
                                 "sigma_loss", "sigma_target", "sigma_grad"])
    parser.add_argument("--facet-by", default="",
                        help="Comma-separated fields. One subplot per combination.")
    parser.add_argument("--hue-by", default="",
                        help="Comma-separated fields. Distinguishes curves within a subplot. "
                             "Defaults to varying fields not in --facet-by (minus 'seed' when --agg-seed).")
    parser.add_argument("--agg-seed", action="store_true",
                        help="Aggregate (mean + min/max) across seeds within each hue group.")
    parser.add_argument("--col-wrap", type=int, default=3,
                        help="Max subplots per row when faceting. Default 3.")
    parser.add_argument("--figsize", default=None,
                        help="Override figsize, e.g. '12,8'")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default="round")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--out", default=None,
                        help="Output image path. If omitted, shows interactively.")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if args.out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filters = {
        "dataset": args.dataset,
        "partition": args.partition,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
    }
    sigma_filter = set(str(s) for s in args.sigma_lr)
    method_filter = set(args.method)

    runs = load_runs(args.log_dir)
    if not runs:
        print(f"No CSV files found under {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    selected = []
    for meta, rows in runs:
        key = parse_run_key(rows)
        if not passes_filters(key, filters):
            continue
        if sigma_filter:
            raw = str(key.get("sigma_lr_init", ""))
            try:
                numeric = f"{float(raw)}"
            except ValueError:
                numeric = ""
            if raw not in sigma_filter and numeric not in sigma_filter:
                continue
        if method_filter:
            m1 = str(key.get("method", ""))
            m2 = str(key.get("sigma_mode", ""))
            if m1 not in method_filter and m2 not in method_filter:
                continue
        selected.append((meta, rows, key))

    if not selected:
        print("No runs matched the filters.", file=sys.stderr)
        sys.exit(2)

    # Which fields actually vary across the selected set
    varying = [f for f in FIELDS
               if len({str(k.get(f, "")) for _, _, k in selected}) > 1]

    if args.list_only:
        for meta, rows, key in selected:
            print(f"{meta['_run_tag']}  -> "
                  + ", ".join(f"{f}={key.get(f,'')}" for f in FIELDS))
        print(f"\n{len(selected)} run(s) matched.")
        return

    facet_fields = split_fields(args.facet_by)
    for f in facet_fields:
        if f not in FIELDS:
            print(f"Unknown --facet-by field: {f}. Valid: {FIELDS}", file=sys.stderr)
            sys.exit(3)

    hue_fields = split_fields(args.hue_by)
    if not hue_fields:
        hue_fields = [f for f in varying
                      if f not in facet_fields and not (args.agg_seed and f == "seed")]

    # Bucket by facet combination
    facets = defaultdict(list)
    for meta, rows, key in selected:
        fc = tuple(str(key.get(f, "")) for f in facet_fields) if facet_fields else ("all",)
        facets[fc].append((meta, rows, key))

    facet_keys = sorted(facets.keys())
    n = len(facet_keys)
    ncols = min(args.col_wrap, n) if n > 0 else 1
    nrows = (n + ncols - 1) // ncols

    if args.figsize:
        fw, fh = (float(x) for x in args.figsize.split(","))
    else:
        fw, fh = max(5.0 * ncols, 6.0), max(3.5 * nrows, 4.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh),
                             squeeze=False, sharex=True)

    for idx, fc in enumerate(facet_keys):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        subset = facets[fc]

        # Group curves inside this subplot by hue_fields
        hue_groups = defaultdict(list)
        for meta, rows, key in subset:
            h_key = tuple(str(key.get(f, "")) for f in hue_fields) if hue_fields else ("run",)
            hue_groups[h_key].append((rows, key))

        for h_key, items in sorted(hue_groups.items()):
            label = fmt_combo(hue_fields, h_key) if hue_fields else "run"
            if args.agg_seed and len(items) > 1:
                curves = []
                for rows, _ in items:
                    xs, ys = extract_metric(rows, args.metric)
                    if len(xs) > 0:
                        curves.append((xs, ys))
                if not curves:
                    continue
                min_len = min(len(ys) for _, ys in curves)
                stacked = np.stack([ys[:min_len] for _, ys in curves], axis=0)
                xs0 = curves[0][0][:min_len]
                mean = stacked.mean(axis=0)
                lo = stacked.min(axis=0)
                hi = stacked.max(axis=0)
                line, = ax.plot(xs0, mean, label=f"{label} (n={len(curves)})")
                ax.fill_between(xs0, lo, hi, alpha=0.15, color=line.get_color())
            else:
                for rows, key in items:
                    xs, ys = extract_metric(rows, args.metric)
                    if len(xs) == 0:
                        continue
                    this_label = label
                    if not args.agg_seed and "seed" not in hue_fields and "seed" in varying:
                        this_label = f"{label}, seed={key.get('seed', '')}"
                    ax.plot(xs, ys, label=this_label, alpha=0.85)

        ax.set_title(fmt_combo(facet_fields, fc) if facet_fields else "")
        ax.grid(True, alpha=0.3)
        if args.log_y:
            ax.set_yscale("log")
        if r == nrows - 1:
            ax.set_xlabel(args.xlabel)
        if c == 0:
            ax.set_ylabel(args.ylabel or args.metric)
        ax.legend(fontsize=7, loc="best")

    # Hide any unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    sup = args.title or f"{args.metric} ({len(selected)} runs)"
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
