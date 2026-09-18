#!/bin/bash
set -euo pipefail

if [[ -z "${NOHUP_WRAPPED:-}" ]]; then
    export NOHUP_WRAPPED=1

    OUTER_LOG="/tmp/$(basename "$0").nohup.log"

    nohup "$0" "$@" \
        > "$OUTER_LOG" 2>&1 &

    echo "[INFO] Script detached with nohup (PID=$!)."
    echo "[INFO] Outer log: $OUTER_LOG"
    exit 0
fi


REPO_ROOT="$(cd "${REPO_ROOT:-.}" && pwd)"

WORKDIR="${WORKDIR:-${REPO_ROOT}/src/r1-vl/src/open_r1/rollout_grpo}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-grpo_reasoning_paper_aligned.py}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${REPO_ROOT}/checkpoints/sft_reason}"

DATASET_REASONING="${DATASET_REASONING:-${REPO_ROOT}/data/reason_data_grpo.json}"


CORE_NAME="${CORE_NAME:-medspa_grpo_reason}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-${EPOCHS:-2}}"

DATE_TAG="$(date '+%d%m%y')"
TIME_TAG="$(date '+%H%M%S')"

RUN_NAME="${RUN_NAME:-${CORE_NAME}_${NUM_TRAIN_EPOCHS}e_${DATE_TAG}_${TIME_TAG}}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/grpo/reason/${RUN_NAME}}"

mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/train_reasoning.log"


export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

export CHEXBERT_DEVICE="${CHEXBERT_DEVICE:-rank}"
export RADGRAPH_DEVICE="${RADGRAPH_DEVICE:-cpu}"


TRAINING_GPUS="${TRAINING_GPUS:-0,1,2,3,4,5,6,7}"

IFS=',' read -r -a GPU_LIST <<< "$TRAINING_GPUS"
NPROC="${NPROC:-${#GPU_LIST[@]}}"

export CUDA_VISIBLE_DEVICES="$TRAINING_GPUS"


if [[ ! -f "${WORKDIR}/${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] Training script not found."
    exit 1
fi

if [[ ! -f "$DATASET_REASONING" ]]; then
    echo "[ERROR] Training dataset not found."
    exit 1
fi

if [[ ! -d "$SFT_ADAPTER_PATH" ]]; then
    echo "[ERROR] SFT adapter not found."
    exit 1
fi


echo "[INFO] Starting MedSPA PR R-Align training."
echo "[INFO] Number of processes: ${NPROC}"


cd "$WORKDIR"

set +e

torchrun \
    --nproc_per_node="$NPROC" \
    "$TRAIN_SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$BASE_MODEL" \
    --dataset_name "$DATASET_REASONING" \
    --num_generations "${NUM_GENERATIONS:-4}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
    --learning_rate "${LEARNING_RATE:-2e-5}" \
    --max_prompt_length "${MAX_PROMPT_LEN:-512}" \
    --max_completion_length "${MAX_COMPLETION_LEN:-256}" \
    --max_rollout_steps "${MAX_ROLLOUT_STEPS:-10}" \
    --coverage_beta "${COVERAGE_BETA:-0.5}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top_p "${TOP_P:-1.0}" \
    --logging_steps "${LOGGING_STEPS:-50}" \
    --dtype "bfloat16" \
    --tf32 "True" \
    --report_to "none" \
    --gradient_checkpointing "true" \
    --attn_implementation "${ATTN_IMPL:-sdpa}" \
    --target_size "${TARGET_SIZE:-768}" \
    --min_pixels "${MIN_PIXELS:-3136}" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-100}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --save_only_model "true" \
    --trust_remote_code "True" \
    --use_peft "True" \
    --sft_adapter_path "$SFT_ADAPTER_PATH" \
    --sft_adapter_name "${SFT_ADAPTER_NAME:-sft}" \
    --grpo_adapter_name "${GRPO_ADAPTER_NAME:-grpo}" \
    > "$LOGFILE" 2>&1

EXIT_CODE=$?

set -e


if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "[INFO] GRPO reasoning training completed successfully."
else
    echo "[ERROR] GRPO reasoning training failed with exit code ${EXIT_CODE}."
fi

echo "[INFO] Training log saved."

exit "$EXIT_CODE"