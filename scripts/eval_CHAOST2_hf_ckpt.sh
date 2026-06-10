#!/usr/bin/env bash
# GPUID1=0 bash scripts/eval_CHAOST2_hf_ckpt.sh
set -euo pipefail

GPUID1=${GPUID1:-0}
export CUDA_VISIBLE_DEVICES=$GPUID1

DATASET='CHAOST2'
NWORKER=4
ALL_EV=(0 1 2 3 4)
TEST_LABEL=[1,2,3,4]

NSTEP=39001
DECAY=0.98
MAX_ITER=3000
SNAPSHOT_INTERVAL=3000
SEED=2025
N_PART=3
ALL_SUPP=(2)

LOGDIR="./results_eval_chaost2"
WEIGHT_ROOT="./checkpoints/fobsam_pretrained/CHAOST2/exps_train_on_CHAOST2_FSMIS_FoB"

mkdir -p "$LOGDIR"

for EVAL_FOLD in "${ALL_EV[@]}"; do
  for SUPP_IDX in "${ALL_SUPP[@]}"; do
    RELOAD_MODEL_PATH="${WEIGHT_ROOT}/FSMIS_train_CHAOST2_cv${EVAL_FOLD}/ckpt.pth"

    if [ ! -f "$RELOAD_MODEL_PATH" ]; then
      echo "ERROR: checkpoint not found:"
      echo "$RELOAD_MODEL_PATH"
      echo "Available:"
      find "$WEIGHT_ROOT" -type f -name "ckpt.pth" | sort
      exit 1
    fi

    echo "============================================================"
    echo "Evaluating CHAOST2 fold ${EVAL_FOLD}"
    echo "Checkpoint: ${RELOAD_MODEL_PATH}"

    python3 test.py with \
      mode="test" \
      gpu_id=0 \
      dataset=$DATASET \
      num_workers=$NWORKER \
      n_steps=$NSTEP \
      eval_fold=$EVAL_FOLD \
      max_iters_per_load=$MAX_ITER \
      supp_idx=$SUPP_IDX \
      test_label=$TEST_LABEL \
      seed=$SEED \
      n_part=$N_PART \
      reload_model_path="$RELOAD_MODEL_PATH" \
      save_snapshot_every=$SNAPSHOT_INTERVAL \
      lr_step_gamma=$DECAY \
      path.log_dir=$LOGDIR
  done
done
