#!/usr/bin/env bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT"

echo "=========================================="
echo " Training all methods for comparison"
echo "=========================================="

for config in \
    src/uncertainty_nav/config/train_ensemble.yaml \
    src/uncertainty_nav/config/train_vanilla.yaml \
    src/uncertainty_nav/config/train_lstm.yaml \
    src/uncertainty_nav/config/train_gru.yaml \
    src/uncertainty_nav/config/train_large_mlp.yaml; do

    policy=$(python3 -c "import yaml; d=yaml.safe_load(open('$config')); print(d['policy_type'])")
    echo ""
    echo "--- Training: $policy ---"
    python3 scripts/train/ppo_trainer.py "$config"
done

echo ""
echo "All training complete:"
