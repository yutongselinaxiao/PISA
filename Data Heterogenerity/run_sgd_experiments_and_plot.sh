#!/bin/bash
# Run the SGD-epochs experiment sweep, wait for it to finish, then
# generate plots from the locally saved CSV metrics.
#
# The experiment launcher (generate_and_run_sisa_jobs_sgd_epochs.py) already
# spawns all per-run shell scripts in parallel across GPUs and waits for
# all of them to finish before returning, so we just chain the plot step
# after it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Must match OUTPUT_DIR / LOCAL_METRICS_DIR in generate_and_run_sisa_jobs_sgd_epochs.py
RUN_ROOT="generated_sisa_sgd_epochs_runs_updated_local_update"
METRICS_DIR="${RUN_ROOT}/local_metrics"
PLOT_DIR="${RUN_ROOT}/plots"

mkdir -p "$PLOT_DIR"

echo "=============================================="
echo "[1/2] Running SGD-epochs experiment sweep..."
echo "=============================================="
python generate_and_run_sisa_jobs_sgd_epochs.py

echo
echo "=============================================="
echo "[2/2] Generating plots from ${METRICS_DIR}"
echo "=============================================="

if [ ! -d "$METRICS_DIR" ] || [ -z "$(ls -A "$METRICS_DIR" 2>/dev/null)" ]; then
    echo "ERROR: no local CSVs found under ${METRICS_DIR}" >&2
    echo "Did the experiment runs produce any output?" >&2
    exit 1
fi

# Test accuracy: one subplot per (dataset, partition), colored by local lr,
# averaged over seeds.
python plot_sgd_experiments.py \
    --log-dir "$METRICS_DIR" \
    --method sgd_adaptive --method online_convex_bal \
    --facet-by dataset,partition --hue-by lr,epochs \
    --agg-seed --metric test_acc \
    --title "Test accuracy — sgd_adaptive (mean over seeds)" \
    --out "${PLOT_DIR}/test_acc_adaptive_facet_dataset_partition.png"

# Primal residual on log-y, same facet layout.
python plot_sgd_experiments.py \
    --log-dir "$METRICS_DIR" \
    --method sgd_adaptive --method online_convex_bal \
    --facet-by dataset,partition --hue-by lr,epochs \
    --agg-seed --metric primal_res_avg --log-y \
    --title "Primal residual — sgd_adaptive (mean over seeds)" \
    --out "${PLOT_DIR}/primal_res_adaptive_facet_dataset_partition.png"

# Sigma trajectory (log scale), to inspect adaptive sigma behavior.
python plot_sgd_experiments.py \
    --log-dir "$METRICS_DIR" \
    --method sgd_adaptive --method online_convex_bal \
    --facet-by dataset,partition --hue-by sigma_lr_init,lr \
    --agg-seed --metric sigma_value --log-y \
    --title "Sigma trajectory — sgd_adaptive (mean over seeds)" \
    --out "${PLOT_DIR}/sigma_value_adaptive_facet_dataset_partition.png"

# Per-lr view: one subplot per local lr, curves colored by initial sigma.
python plot_sgd_experiments.py \
    --log-dir "$METRICS_DIR" \
    --method sgd_adaptive --method online_convex_bal \
    --facet-by lr --hue-by sigma_lr_init,epochs \
    --agg-seed --metric test_acc \
    --title "Test accuracy by local lr — sgd_adaptive" \
    --out "${PLOT_DIR}/test_acc_adaptive_facet_lr.png"

# Baseline (original) comparison: adaptive vs. original on test accuracy.
python plot_sgd_experiments.py \
    --log-dir "$METRICS_DIR" \
    --facet-by dataset,partition --hue-by method,sigma_lr_init \
    --agg-seed --metric test_acc \
    --title "Adaptive vs. original — test accuracy (mean over seeds)" \
    --out "${PLOT_DIR}/test_acc_adaptive_vs_original.png"

echo
echo "Done. Plots written to ${PLOT_DIR}/"
ls -lh "${PLOT_DIR}"
