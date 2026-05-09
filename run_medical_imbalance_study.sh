#!/bin/bash

source venv/bin/activate

RATIOS=(1 10 50)
SEEDS=(42 7 123)

mkdir -p study_logs

for RATIO in "${RATIOS[@]}"
do
  for SEED in "${SEEDS[@]}"
  do
    echo "===================================================="
    echo "Running ETF | ratio=${RATIO} | seed=${SEED}"
    echo "===================================================="

    python train.py \
      --profile cpu_debug \
      --override \
      dataset.name=ham10000 \
      dataset.imbalance_ratio=${RATIO} \
      model.head=etf \
      model.pretrained=true \
      training.loss=focal \
      training.epochs=30 \
      seed=${SEED} \
      nc_regularization.collapse_weight=0.01 \
      tracking.tensorboard=false \
      visualization.enabled=false \
      analysis.lightweight=true \
      > study_logs/etf_r${RATIO}_s${SEED}.log 2>&1

    echo "Completed ratio=${RATIO} seed=${SEED}"
  done
done

echo "===================================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "===================================================="
