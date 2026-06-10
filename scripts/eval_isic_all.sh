#!/usr/bin/env bash
# Usage:
#   GPUID1=0 SETTINGS="1" FOLDS1="1" bash scripts/eval_isic_all.sh
#   GPUID1=0 SETTINGS="1 2" bash scripts/eval_isic_all.sh

set -euo pipefail

cd "$(dirname "$0")/.."

GPUID1=${GPUID1:-0}
export CUDA_VISIBLE_DEVICES="${GPUID1}"

SETTINGS=${SETTINGS:-"1 2"}
FOLDS1=${FOLDS1:-"1 2 3 4 5"}
FOLDS2=${FOLDS2:-"1 2 3"}
SUPPS=${SUPPS:-"2"}

NWORKER=${NWORKER:-4}
NSTEP=${NSTEP:-39001}
DECAY=${DECAY:-0.98}
MAX_ITER=${MAX_ITER:-3000}
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-3000}
SEED=${SEED:-2025}
N_PART=${N_PART:-3}
CKPT_STEP=${CKPT_STEP:-39000}

mkdir -p logs_isic_eval

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
    echo "Please download missing common weights first."
    exit 1
  fi
}

find_setting1_ckpt() {
  local fold="$1"
  local root="./exps_train_on_isic_setting1_FSMIS_FoB"
  local ckpt=""

  ckpt=$(
    for d in \
      "$root/FSMIS_train_isic_setting1_cv${fold}" \
      "$root/FSMIS_train_isic_cv${fold}"
    do
      [ -d "$d" ] && find "$d" -type f -path "*/snapshots/${CKPT_STEP}.pth"
    done | sort -V | tail -n 1
  )

  if [ -z "$ckpt" ]; then
    ckpt=$(
      for d in \
        "$root/FSMIS_train_isic_setting1_cv${fold}" \
        "$root/FSMIS_train_isic_cv${fold}"
      do
        [ -d "$d" ] && find "$d" -type f -path "*/snapshots/*.pth"
      done | sort -V | tail -n 1
    )
  fi

  echo "$ckpt"
}

find_setting2_ckpt() {
  local fold="$1"
  local root="./exps_train_on_isic_FSMIS_FoB"
  local ckpt=""

  ckpt=$(
    for d in \
      "$root/FSMIS_train_isic_setting2_cv${fold}" \
      "$root/FSMIS_train_isic_cv${fold}"
    do
      [ -d "$d" ] && find "$d" -type f -path "*/snapshots/${CKPT_STEP}.pth"
    done | sort -V | tail -n 1
  )

  if [ -z "$ckpt" ]; then
    ckpt=$(
      for d in \
        "$root/FSMIS_train_isic_setting2_cv${fold}" \
        "$root/FSMIS_train_isic_cv${fold}"
      do
        [ -d "$d" ] && find "$d" -type f -path "*/snapshots/*.pth"
      done | sort -V | tail -n 1
    )
  fi

  echo "$ckpt"
}

eval_setting1() {
  read -r -a FOLD_LIST <<< "${FOLDS1}"
  read -r -a SUPP_LIST <<< "${SUPPS}"

  local LOGDIR="./results_eval_isic_setting1"
  mkdir -p "$LOGDIR"

  for EVAL_FOLD in "${FOLD_LIST[@]}"; do
    for SUPP_IDX in "${SUPP_LIST[@]}"; do
      RELOAD_MODEL_PATH="$(find_setting1_ckpt "$EVAL_FOLD")"

      if [ -z "$RELOAD_MODEL_PATH" ] || [ ! -f "$RELOAD_MODEL_PATH" ]; then
        echo "ERROR: cannot find Setting I checkpoint for fold ${EVAL_FOLD}"
        echo "Expected under: ./exps_train_on_isic_setting1_FSMIS_FoB"
        find ./exps_train_on_isic_setting1_FSMIS_FoB -type f -name "*.pth" 2>/dev/null | sort || true
        exit 1
      fi

      echo "============================================================"
      echo "Evaluating ISIC Setting I fold ${EVAL_FOLD}"
      echo "Checkpoint: ${RELOAD_MODEL_PATH}"
      echo "============================================================"

      python3 test.py with \
        mode="test" \
        gpu_id=0 \
        dataset="isic" \
        isic_setting=1 \
        isic_setting_1_base_path="data/isic/ISIC_setting_1" \
        num_workers="${NWORKER}" \
        n_steps="${NSTEP}" \
        eval_fold="${EVAL_FOLD}" \
        max_iters_per_load="${MAX_ITER}" \
        supp_idx="${SUPP_IDX}" \
        test_label=[] \
        seed="${SEED}" \
        n_part="${N_PART}" \
        reload_model_path="${RELOAD_MODEL_PATH}" \
        save_snapshot_every="${SNAPSHOT_INTERVAL}" \
        lr_step_gamma="${DECAY}" \
        path.log_dir="${LOGDIR}" \
        2>&1 | tee "logs_isic_eval/eval_setting1_cv${EVAL_FOLD}_supp${SUPP_IDX}.log"
    done
  done
}

eval_setting2() {
  read -r -a FOLD_LIST <<< "${FOLDS2}"
  read -r -a SUPP_LIST <<< "${SUPPS}"

  local LOGDIR="./results_eval_isic_setting2"
  mkdir -p "$LOGDIR"

  for EVAL_FOLD in "${FOLD_LIST[@]}"; do
    for SUPP_IDX in "${SUPP_LIST[@]}"; do
      RELOAD_MODEL_PATH="$(find_setting2_ckpt "$EVAL_FOLD")"

      if [ -z "$RELOAD_MODEL_PATH" ] || [ ! -f "$RELOAD_MODEL_PATH" ]; then
        echo "ERROR: cannot find Setting II checkpoint for fold ${EVAL_FOLD}"
        echo "Expected under: ./exps_train_on_isic_FSMIS_FoB"
        find ./exps_train_on_isic_FSMIS_FoB -type f -name "*.pth" 2>/dev/null | sort || true
        exit 1
      fi

      echo "============================================================"
      echo "Evaluating ISIC Setting II fold ${EVAL_FOLD}"
      echo "Checkpoint: ${RELOAD_MODEL_PATH}"
      echo "============================================================"

      python3 test.py with \
        mode="test" \
        gpu_id=0 \
        dataset="isic" \
        isic_setting=2 \
        isic_setting_2_base_path="data/isic/combine" \
        num_workers="${NWORKER}" \
        n_steps="${NSTEP}" \
        eval_fold="${EVAL_FOLD}" \
        max_iters_per_load="${MAX_ITER}" \
        supp_idx="${SUPP_IDX}" \
        test_label=[] \
        seed="${SEED}" \
        n_part="${N_PART}" \
        reload_model_path="${RELOAD_MODEL_PATH}" \
        save_snapshot_every="${SNAPSHOT_INTERVAL}" \
        lr_step_gamma="${DECAY}" \
        path.log_dir="${LOGDIR}" \
        2>&1 | tee "logs_isic_eval/eval_setting2_cv${EVAL_FOLD}_supp${SUPP_IDX}.log"
    done
  done
}

patch_weight_paths
check_common_weights

read -r -a SETTING_LIST <<< "${SETTINGS}"

for SETTING in "${SETTING_LIST[@]}"; do
  if [ "$SETTING" = "1" ]; then
    eval_setting1
  elif [ "$SETTING" = "2" ]; then
    eval_setting2
  else
    echo "ERROR: unknown setting: $SETTING"
    echo "Allowed: 1 2"
    exit 1
  fi
done

echo "ISIC evaluation finished."
echo "Setting I logs:  ./results_eval_isic_setting1"
echo "Setting II logs: ./results_eval_isic_setting2"
