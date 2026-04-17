#!/bin/bash
# =============================================================
# Revision: Complete evaluation pipeline
# =============================================================

set -e
echo "=============================================="
echo " Revision Evaluation Pipeline"
echo "=============================================="

# Ensure package is importable
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/uncertainty_nav"

# 1: Evaluate all policies on all environments (A, B, C, D)
echo ""
echo ">>> Step 1: All policies x All environments (A, B, C, D)"
echo "    This evaluates 5 policies x 4 envs = 20 configurations"
echo "    200 episodes x 5 seeds each"
echo ""
python3 scripts/eval/evaluate_all_envs.py --all --n_episodes 200 --n_seeds 5

# 2: Env E noise sweep on Env B layout
echo ""
echo ">>> Step 2: Env E noise sweep (Pillar 2)"
echo "    Ensemble + Vanilla + Large MLP across 8 noise levels"
echo ""
python3 scripts/eval/evaluate_all_envs.py --env_e

# 3: AUROC by ensemble size, third revision
echo ""
echo ">>> Step 3: AUROC by ensemble size"
echo "    N=1,2,3,5,10 with AUROC and ECE metrics"
echo ""
python3 scripts/eval/evaluate_all_envs.py --auroc_ablation

# 4: LSTM diagnostic,  shorter training runs
echo ""
echo ">>> Step 4: LSTM diagnostic (300K steps each)"
echo "    Tests BPTT truncation hypothesis"
echo ""
python3 scripts/train/lstm_diagnostic.py

echo ""
echo "=============================================="
echo " All revision evaluations complete!"
echo " Results saved to experiments/results/"
echo "=============================================="
echo ""
echo "Output files:"
echo "  experiments/results/{policy}_env{A,B,C,D}.json"
echo "  experiments/results/env_e_noise_sweep.json"
echo "  experiments/results/ensemble_size_auroc.json"
echo "  experiments/results/lstm_diagnostic.json"
