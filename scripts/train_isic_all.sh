#!/usr/bin/env bash
# Usage:
#   GPUID1=0 SETTINGS="1" FOLDS1="1" bash scripts/train_isic_all.sh
#   GPUID1=0 SETTINGS="1 2" bash scripts/train_isic_all.sh

set -euo pipefail

cd "$(dirname "$0")/.."

GPUID1=${GPUID1:-0}
export CUDA_VISIBLE_DEVICES="${GPUID1}"

SETTINGS=${SETTINGS:-"1 2"}
FOLDS1=${FOLDS1:-"1 2 3 4 5"}
FOLDS2=${FOLDS2:-"1 2 3"}

NWORKER=${NWORKER:-4}
NSTEP=${NSTEP:-39001}
DECAY=${DECAY:-0.98}
MAX_ITER=${MAX_ITER:-3000}
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-1000}
SEED=${SEED:-2025}

mkdir -p logs_isic_train

patch_weight_paths() {
  python - <<'PY'
from pathlib import Path

repls = {
    '".../sam_vit_h_4b8939.pth"': '"checkpoints/sam_vit_h_4b8939.pth"',
    '".../resnet101-63fe2227.pth"': '"checkpoints/resnet101-63fe2227.pth"',
    '".../deeplabv3_resnet101_coco-586e9e4e.pth"': '"checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth"',
}

for fname in ["test.py", "SAM.py", "models/encoder.py"]:
    p = Path(fname)
    if not p.exists():
        continue
    s = p.read_text()
    for old, new in repls.items():
        s = s.replace(old, new)
    p.write_text(s)

print("weight paths patched")
PY
}

check_common_weights() {
  local missing=0

  for f in \
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
    echo "Please download missing common weights first."
    exit 1
  fi
}

check_isic_data() {
  echo "============================================================"
  echo "Checking ISIC data"
  echo "============================================================"

  if [ ! -d "data/isic/ISIC_setting_1/images" ]; then
    echo "ERROR: missing data/isic/ISIC_setting_1/images"
    echo "Run: bash scripts/prepare_isic_data.sh"
    exit 1
  fi

  if [ ! -d "data/isic/ISIC_setting_1/superpixels" ]; then
    echo "ERROR: missing data/isic/ISIC_setting_1/superpixels"
    echo "Run: bash scripts/prepare_isic_data.sh"
    exit 1
  fi

  if [ ! -d "data/isic/combine" ]; then
    echo "ERROR: missing data/isic/combine"
    echo "Run: bash scripts/prepare_isic_data.sh"
    exit 1
  fi

  echo "Setting I images:"
  find data/isic/ISIC_setting_1/images -maxdepth 1 -name "*.jpg" | wc -l

  echo "Setting I superpixels:"
  find data/isic/ISIC_setting_1/superpixels -maxdepth 1 -name "*_mask.png" | wc -l

  echo "Setting II class folders:"
  find data/isic/combine/ISIC2018_Task1-2_Training_Input/1 -maxdepth 1 -name "*.jpg" | wc -l
  find data/isic/combine/ISIC2018_Task1-2_Training_Input/2 -maxdepth 1 -name "*.jpg" | wc -l
  find data/isic/combine/ISIC2018_Task1-2_Training_Input/3 -maxdepth 1 -name "*.jpg" | wc -l
}

train_setting1() {
  read -r -a FOLD_LIST <<< "${FOLDS1}"

  local LOGDIR="./exps_train_on_isic_setting1_FSMIS_FoB"
  mkdir -p "$LOGDIR"

  for EVAL_FOLD in "${FOLD_LIST[@]}"; do
    echo "============================================================"
    echo "Training ISIC Setting I fold ${EVAL_FOLD}"
    echo "Logdir: ${LOGDIR}"
    echo "============================================================"

    python3 train.py with \
      mode="train" \
      gpu_id=0 \
      dataset="isic" \
      isic_setting=1 \
      isic_setting_1_base_path="data/isic/ISIC_setting_1" \
      num_workers="${NWORKER}" \
      n_steps="${NSTEP}" \
      eval_fold="${EVAL_FOLD}" \
      test_label=None \
      exclude_label=None \
      use_gt=False \
      max_iters_per_load="${MAX_ITER}" \
      seed="${SEED}" \
      save_snapshot_every="${SNAPSHOT_INTERVAL}" \
      lr_step_gamma="${DECAY}" \
      path.log_dir="${LOGDIR}" \
      2>&1 | tee "logs_isic_train/train_setting1_cv${EVAL_FOLD}.log"
  done
}

train_setting2() {
  read -r -a FOLD_LIST <<< "${FOLDS2}"

  local LOGDIR="./exps_train_on_isic_FSMIS_FoB"
  mkdir -p "$LOGDIR"

  for EVAL_FOLD in "${FOLD_LIST[@]}"; do
    echo "============================================================"
    echo "Training ISIC Setting II fold ${EVAL_FOLD}"
    echo "Logdir: ${LOGDIR}"
    echo "============================================================"

    python3 train.py with \
      mode="train" \
      gpu_id=0 \
      dataset="isic" \
      isic_setting=2 \
      isic_setting_2_base_path="data/isic/combine" \
      num_workers="${NWORKER}" \
      n_steps="${NSTEP}" \
      eval_fold="${EVAL_FOLD}" \
      test_label=None \
      exclude_label=None \
      use_gt=False \
      max_iters_per_load="${MAX_ITER}" \
      seed="${SEED}" \
      save_snapshot_every="${SNAPSHOT_INTERVAL}" \
      lr_step_gamma="${DECAY}" \
      path.log_dir="${LOGDIR}" \
      2>&1 | tee "logs_isic_train/train_setting2_cv${EVAL_FOLD}.log"
  done
}

patch_weight_paths
check_common_weights
check_isic_data

read -r -a SETTING_LIST <<< "${SETTINGS}"

for SETTING in "${SETTING_LIST[@]}"; do
  if [ "$SETTING" = "1" ]; then
    train_setting1
  elif [ "$SETTING" = "2" ]; then
    train_setting2
  else
    echo "ERROR: unknown setting: $SETTING"
    echo "Allowed: 1 2"
    exit 1
  fi
done

echo "ISIC training finished."
