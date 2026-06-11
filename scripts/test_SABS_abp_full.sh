#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATASET='SABS'
NWORKER=${NWORKER:-16}
ALL_EV=(0 1 2 3 4)
ALL_SUPP=(2)

TEST_LABEL=[1,2,3,6]

NSTEP=39001
DECAY=0.98
MAX_ITER=3000
SNAPSHOT_INTERVAL=3000
SEED=2025
N_PART=3

TRAIN_LOGDIR=./exps_train_on_SABS_ABP_FULL
RESULT_LOGDIR=./results_SABS_ABP_FULL

mkdir -p ${RESULT_LOGDIR}

export USE_MULTIRING_BPPC=1
export USE_LEARNABLE_RING_FUSION=1
export USE_PROMPT_VALIDITY_FILTER=1
export USE_SAM_SELECTOR=1

export NEG_FG_THRESHOLD=${NEG_FG_THRESHOLD:-0.85}
export NEG_MIN_DIST=${NEG_MIN_DIST:-5.0}
export NEG_TOPK=${NEG_TOPK:-256}
export NEG_MIN_KEEP=${NEG_MIN_KEEP:-6}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SAM_CKPT=${SAM_CKPT}"
echo "TRAIN_LOGDIR=${TRAIN_LOGDIR}"
echo "RESULT_LOGDIR=${RESULT_LOGDIR}"

for EVAL_FOLD in "${ALL_EV[@]}"
do
    echo "========================================================================"
    echo "Testing fold ${EVAL_FOLD}"

    RELOAD_MODEL_PATH=$(find ${TRAIN_LOGDIR}/FSMIS_train_${DATASET}_cv${EVAL_FOLD} -name "39000.pth" | sort -V | tail -n 1)

    if [ -z "${RELOAD_MODEL_PATH}" ]; then
        echo "No checkpoint found for fold ${EVAL_FOLD}"
        exit 1
    fi

    echo "Using checkpoint: ${RELOAD_MODEL_PATH}"

    for SUPP_IDX in "${ALL_SUPP[@]}"
    do
        python3 test.py with \
            mode="test" \
            dataset=${DATASET} \
            num_workers=${NWORKER} \
            n_steps=${NSTEP} \
            eval_fold=${EVAL_FOLD} \
            max_iters_per_load=${MAX_ITER} \
            supp_idx=${SUPP_IDX} \
            test_label=${TEST_LABEL} \
            seed=${SEED} \
            n_part=${N_PART} \
            reload_model_path=${RELOAD_MODEL_PATH} \
            save_snapshot_every=${SNAPSHOT_INTERVAL} \
            lr_step_gamma=${DECAY} \
            path.log_dir=${RESULT_LOGDIR}
    done
done
