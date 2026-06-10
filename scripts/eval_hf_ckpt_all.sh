#!/usr/bin/env bash
# Usage:
#   GPUID1=0 bash scripts/eval_hf_ckpt_all.sh
#
# Optional:
#   DATASETS="SABS" GPUID1=0 bash scripts/eval_hf_ckpt_all.sh
#   DATASETS="CHAOST2" GPUID1=0 bash scripts/eval_hf_ckpt_all.sh
#   DATASETS="SABS CHAOST2" FOLDS="0" GPUID1=0 bash scripts/eval_hf_ckpt_all.sh

set -euo pipefail

GPUID1=${GPUID1:-0}
export CUDA_VISIBLE_DEVICES="${GPUID1}"

DATASETS=${DATASETS:-"SABS CHAOST2"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SUPPS=${SUPPS:-"2"}

NWORKER=${NWORKER:-4}
NSTEP=${NSTEP:-39001}
DECAY=${DECAY:-0.98}
MAX_ITER=${MAX_ITER:-3000}
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-3000}
SEED=${SEED:-2025}
N_PART=${N_PART:-3}

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a FOLD_LIST <<< "${FOLDS}"
read -r -a SUPP_LIST <<< "${SUPPS}"

echo "============================================================"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATASETS=${DATASETS}"
echo "FOLDS=${FOLDS}"
echo "SUPPS=${SUPPS}"
echo "============================================================"

check_common_weights() {
  echo "[Check] common pretrained weights"

  local missing=0

  for f in \
    "checkpoints/sam_vit_h_4b8939.pth" \
    "checkpoints/resnet101-63fe2227.pth" \
    "checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth"
  do
    if [ ! -f "$f" ]; then
      echo "ERROR: missing $f"
      missing=1
    else
      echo "OK: $f"
    fi
  done

  if [ "$missing" -ne 0 ]; then
    echo "Please download the missing common weights first."
    exit 1
  fi
}

configure_dataset() {
  local dataset="$1"

  if [ "$dataset" = "SABS" ]; then
    TEST_LABEL='[1,2,3,6]'
    LOGDIR="./results_eval_sabs"
    WEIGHT_ROOT="./checkpoints/fobsam_pretrained/SABS/exps_train_on_SABS_FSMIS_FoB"
    CKPT_PREFIX="FSMIS_train_SABS"
    IMG_DIR="./data/SABS/sabs_CT_normalized"
    EXPECTED_N=30

  elif [ "$dataset" = "CHAOST2" ]; then
    TEST_LABEL='[1,2,3,4]'
    LOGDIR="./results_eval_chaost2"
    WEIGHT_ROOT="./checkpoints/fobsam_pretrained/CHAOST2/exps_train_on_CHAOST2_FSMIS_FoB"
    CKPT_PREFIX="FSMIS_train_CHAOST2"
    IMG_DIR="./data/CHAOST2/chaos_MR_T2_normalized"
    EXPECTED_N=20

  else
    echo "ERROR: unknown dataset: $dataset"
    echo "Allowed: SABS CHAOST2"
    exit 1
  fi
}

check_dataset_files() {
  local dataset="$1"

  echo "------------------------------------------------------------"
  echo "[Check] ${dataset} data"

  if [ ! -d "$IMG_DIR" ]; then
    echo "ERROR: data directory not found: $IMG_DIR"
    exit 1
  fi

  local n_img
  local n_lab
  n_img=$(find "$IMG_DIR" -maxdepth 1 -name "image_*.nii.gz" | wc -l)
  n_lab=$(find "$IMG_DIR" -maxdepth 1 -name "label_*.nii.gz" | wc -l)

  echo "image count: $n_img"
  echo "label count: $n_lab"

  if [ "$n_img" -eq 0 ] || [ "$n_lab" -eq 0 ]; then
    echo "ERROR: no image/label files found in $IMG_DIR"
    exit 1
  fi

  if [ "$n_img" -ne "$EXPECTED_N" ] || [ "$n_lab" -ne "$EXPECTED_N" ]; then
    echo "WARNING: expected ${EXPECTED_N} images and labels for ${dataset}, but got images=${n_img}, labels=${n_lab}"
  fi

  if [ ! -d "$WEIGHT_ROOT" ]; then
    echo "ERROR: weight root not found: $WEIGHT_ROOT"
    exit 1
  fi

  local n_ckpt
  n_ckpt=$(find "$WEIGHT_ROOT" -type f -name "ckpt.pth" | wc -l)
  echo "checkpoint count: $n_ckpt"

  if [ "$n_ckpt" -lt 5 ]; then
    echo "WARNING: expected 5 ckpt.pth files for ${dataset}, but found ${n_ckpt}"
    echo "Available checkpoints:"
    find "$WEIGHT_ROOT" -type f -name "ckpt.pth" | sort
  fi

  mkdir -p "$LOGDIR"
}

run_one_dataset() {
  local dataset="$1"

  configure_dataset "$dataset"
  check_dataset_files "$dataset"

  for EVAL_FOLD in "${FOLD_LIST[@]}"; do
    for SUPP_IDX in "${SUPP_LIST[@]}"; do
      RELOAD_MODEL_PATH="${WEIGHT_ROOT}/${CKPT_PREFIX}_cv${EVAL_FOLD}/ckpt.pth"

      if [ ! -f "$RELOAD_MODEL_PATH" ]; then
        echo "ERROR: checkpoint not found:"
        echo "$RELOAD_MODEL_PATH"
        echo "Available checkpoints:"
        find "$WEIGHT_ROOT" -type f -name "ckpt.pth" | sort
        exit 1
      fi

      echo "============================================================"
      echo "Evaluating dataset: ${dataset}"
      echo "Fold: ${EVAL_FOLD}"
      echo "Support index: ${SUPP_IDX}"
      echo "Checkpoint: ${RELOAD_MODEL_PATH}"
      echo "Log dir: ${LOGDIR}"
      echo "============================================================"

      python3 test.py with \
        mode="test" \
        gpu_id=0 \
        dataset="${dataset}" \
        num_workers="${NWORKER}" \
        n_steps="${NSTEP}" \
        eval_fold="${EVAL_FOLD}" \
        max_iters_per_load="${MAX_ITER}" \
        supp_idx="${SUPP_IDX}" \
        test_label="${TEST_LABEL}" \
        seed="${SEED}" \
        n_part="${N_PART}" \
        reload_model_path="${RELOAD_MODEL_PATH}" \
        save_snapshot_every="${SNAPSHOT_INTERVAL}" \
        lr_step_gamma="${DECAY}" \
        path.log_dir="${LOGDIR}" \
        2>&1 | tee "${LOGDIR}/console_${dataset}_cv${EVAL_FOLD}_supp${SUPP_IDX}.log"
    done
  done
}

check_common_weights

for DATASET in "${DATASET_LIST[@]}"; do
  run_one_dataset "$DATASET"
done

echo "============================================================"
echo "Evaluation finished."
echo "SABS logs: ./results_eval_sabs"
echo "CHAOST2 logs: ./results_eval_chaost2"
echo "============================================================"
