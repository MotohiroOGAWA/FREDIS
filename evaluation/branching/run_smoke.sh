#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
export OMP_NUM_THREADS=2
python evaluation/branching/prepare_smoke.py
python -m fredis.cli train fragment-tree \
  --train-dir data/train_preprocessing/test/fragment_tree_smoke/prepared/train_structures \
  --val-dir data/train_preprocessing/test/fragment_tree_smoke/prepared/validation_structures \
  --output-dir data/train_preprocessing/test/fragment_tree_smoke/training \
  --epochs 12 --batch-size 4 --max-samples 16 --lr 0.001 --warmup-steps 0 \
  --reaction-hidden-dim 32 --reaction-main-adduct-dim 32 \
  --branch-path-threshold 0 --max-fragment-nodes 100 \
  --spectrum-hidden-dim 32 --spectrum-num-layers 1 \
  --validation-interval-steps 4 --validation-fraction 0.5
python evaluation/branching/evaluate_smoke.py
