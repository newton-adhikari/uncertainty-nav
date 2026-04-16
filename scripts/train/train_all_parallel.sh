#!/usr/bin/env bash
# Train all methods in parallel with terminal output.
# Wave 1: 5 ensemble members | Wave 2: 4 baselines | Wave 3: 5 extra members
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

export PARALLEL_TRAIN=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=========================================="
echo " Wave 1: 5 ensemble members (independent seeds)"
echo "=========================================="

PIDS=()
for i in 0 1 2 3 4; do
    cfg="src/uncertainty_nav/config/train_ensemble_m${i}.yaml"
    if [ "$i" -eq 0 ]; then
        cfg="src/uncertainty_nav/config/train_ensemble.yaml"
    fi
    echo "  Launching ensemble_m${i} (seed=$i)"
    python3 -u scripts/train/ppo_trainer.py "$cfg" 2>&1 | tee "logs/train_ensemble_m${i}.log" | sed "s/^/[m${i}] /" &
    PIDS+=($!)
done

echo "  Waiting for 5 ensemble members... (PIDs: ${PIDS[*]})"
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED+1))
done
echo "  Wave 1 done ($FAILED failures)"

echo ""
echo "=========================================="
echo " Wave 2: 4 baselines"
echo "=========================================="

PIDS2=()
for config in \
    src/uncertainty_nav/config/train_vanilla.yaml \
    src/uncertainty_nav/config/train_lstm.yaml \
    src/uncertainty_nav/config/train_gru.yaml \
    src/uncertainty_nav/config/train_large_mlp.yaml; do
    name=$(basename "$config" .yaml | sed 's/train_//')
    echo "  Launching $name"
    python3 -u scripts/train/ppo_trainer.py "$config" 2>&1 | tee "logs/train_${name}.log" | sed "s/^/[${name}] /" &
    PIDS2+=($!)
done

echo "  Waiting for 4 baselines... (PIDs: ${PIDS2[*]})"
FAILED2=0
for pid in "${PIDS2[@]}"; do
    wait "$pid" || FAILED2=$((FAILED2+1))
done
echo "  Wave 2 done ($FAILED2 failures)"

echo ""
echo "=========================================="
echo " Wave 3: 5 extra members for N=10 ablation"
echo "=========================================="

PIDS3=()
for i in 5 6 7 8 9; do
    echo "  Launching ensemble_m${i} (seed=$i)"
    python3 -u scripts/train/ppo_trainer.py \
        "src/uncertainty_nav/config/train_ensemble_m${i}.yaml" 2>&1 | tee "logs/train_ensemble_m${i}.log" | sed "s/^/[m${i}] /" &
    PIDS3+=($!)
done

echo "  Waiting for 5 extra members... (PIDs: ${PIDS3[*]})"
FAILED3=0
for pid in "${PIDS3[@]}"; do
    wait "$pid" || FAILED3=$((FAILED3+1))
done
echo "  Wave 3 done ($FAILED3 failures)"

TOTAL=$((FAILED + FAILED2 + FAILED3))
echo ""
if [ "$TOTAL" -eq 0 ]; then
    echo "All training complete. Run: bash scripts/eval/eval_all.sh"
else
    echo "$TOTAL training(s) failed. Check logs/"
fi
