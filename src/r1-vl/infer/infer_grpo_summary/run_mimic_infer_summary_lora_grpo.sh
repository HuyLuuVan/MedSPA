#!/bin/bash

if [[ -z "${NOHUP_WRAPPED:-}" ]]; then
    export NOHUP_WRAPPED=1

    OUTER_LOG="/tmp/$(basename "$0").nohup.log"

    nohup "$0" "$@" > "$OUTER_LOG" 2>&1 &

    echo "[INFO] Script detached with nohup (PID=$!)."
    echo "[INFO] Outer log: $OUTER_LOG"
    exit 0
fi

set -euo pipefail


REPO_ROOT="$(cd "${REPO_ROOT:-.}" && pwd)"


PY_SCRIPT="${PY_SCRIPT:-${REPO_ROOT}/src/r1-vl/infer/infer_grpo_summary/infer_grpo_summary.py}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"


SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${REPO_ROOT}/checkpoints/sft_summary}"

GRPO_ADAPTER_PATH="${GRPO_ADAPTER_PATH:-${REPO_ROOT}/checkpoints/grpo/summary}"


INPUT_SHARD_DIR="${INPUT_SHARD_DIR:-${REPO_ROOT}/results/mimic/grpo_reason}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/mimic/grpo_summary}"


PREFIX_IN="${PREFIX_IN:-test_reasoning_shard}"
SUFFIX_IN="${SUFFIX_IN:-.json}"

PREFIX_OUT="${PREFIX_OUT:-test_caption_shard}"
SUFFIX_OUT="${SUFFIX_OUT:-.json}"
LOG_SUFFIX="${LOG_SUFFIX:-.log}"


ALL_IMAGES_PATH="${ALL_IMAGES_PATH:-${REPO_ROOT}/dataset/mimic_cxr_all_images.txt}"


NUM_GPUS="${NUM_GPUS:-8}"
NUM_SHARDS="${NUM_SHARDS:-16}"


TEMP="${TEMP:-0.2}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW="${MAX_NEW:-512}"
NUM_BEAMS="${NUM_BEAMS:-1}"


IMG_SIZE="${IMG_SIZE:-1024}"
PAD_COLOR="${PAD_COLOR:-0}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"


SAVE_EVERY="${SAVE_EVERY:-50}"
RESUME="${RESUME:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"


export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"


mkdir -p "$OUTPUT_DIR"


if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "[ERROR] Summary inference script not found: $PY_SCRIPT"
    exit 1
fi

if [[ ! -d "$SFT_ADAPTER_PATH" ]]; then
    echo "[ERROR] SUM SFT adapter not found: $SFT_ADAPTER_PATH"
    exit 1
fi

if [[ ! -d "$GRPO_ADAPTER_PATH" ]]; then
    echo "[ERROR] SUM GRPO adapter not found: $GRPO_ADAPTER_PATH"
    exit 1
fi

if [[ ! -d "$INPUT_SHARD_DIR" ]]; then
    echo "[ERROR] Reasoning shard directory not found: $INPUT_SHARD_DIR"
    exit 1
fi

if [[ ! -f "$ALL_IMAGES_PATH" ]]; then
    echo "[ERROR] Image index not found: $ALL_IMAGES_PATH"
    exit 1
fi


RESUME_ARGS=()

if [[ "$RESUME" -eq 1 ]]; then
    RESUME_ARGS+=(--resume)
    echo "[INFO] Resume enabled."
else
    echo "[INFO] Resume disabled."
fi


echo "[INFO] Starting MedSPA SUM R-Align inference."
echo "[INFO] Script       : $PY_SCRIPT"
echo "[INFO] Base model   : $BASE_MODEL"
echo "[INFO] SUM SFT      : $SFT_ADAPTER_PATH"
echo "[INFO] SUM GRPO     : $GRPO_ADAPTER_PATH"
echo "[INFO] Reasoning dir: $INPUT_SHARD_DIR"
echo "[INFO] Output dir   : $OUTPUT_DIR"
echo "[INFO] GPUs         : $NUM_GPUS"
echo "[INFO] Shards       : $NUM_SHARDS"
echo "[INFO] Image size   : $IMG_SIZE"


FAILED=0
COMPLETED=0


for ((WAVE_START=0; WAVE_START<NUM_SHARDS; WAVE_START+=NUM_GPUS)); do

    declare -a WAVE_PIDS=()
    declare -a WAVE_SHARDS=()

    echo
    echo "[INFO] Launching wave starting at shard ${WAVE_START}..."

    for ((GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++)); do

        SHARD=$((WAVE_START + GPU_ID))

        if [[ "$SHARD" -ge "$NUM_SHARDS" ]]; then
            break
        fi

        BASE_IN="${PREFIX_IN}${SHARD}"
        BASE_OUT="${PREFIX_OUT}${SHARD}"

        IN_JSON="${INPUT_SHARD_DIR}/${BASE_IN}${SUFFIX_IN}"
        OUT_JSON="${OUTPUT_DIR}/${BASE_OUT}${SUFFIX_OUT}"
        LOG_FILE="${OUTPUT_DIR}/${BASE_OUT}${LOG_SUFFIX}"

        if [[ ! -f "$IN_JSON" ]]; then
            echo "[ERROR] Missing reasoning shard: $IN_JSON"
            FAILED=$((FAILED + 1))
            continue
        fi

        echo "[GPU ${GPU_ID}] shard=${SHARD}"
        echo "  input : ${IN_JSON}"
        echo "  output: ${OUT_JSON}"
        echo "  log   : ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        "$PYTHON_BIN" "$PY_SCRIPT" \
            --base_model "$BASE_MODEL" \
            --sft_adapter_path "$SFT_ADAPTER_PATH" \
            --grpo_adapter_path "$GRPO_ADAPTER_PATH" \
            --input_json "$IN_JSON" \
            --all_images_path "$ALL_IMAGES_PATH" \
            --output_json "$OUT_JSON" \
            --temperature "$TEMP" \
            --top_p "$TOP_P" \
            --max_new_tokens "$MAX_NEW" \
            --num_beams "$NUM_BEAMS" \
            --save_every "$SAVE_EVERY" \
            --attn_implementation "$ATTN_IMPL" \
            --img_size "$IMG_SIZE" \
            --pad_color "$PAD_COLOR" \
            --min_pixels "$MIN_PIXELS" \
            "${RESUME_ARGS[@]}" \
            > "$LOG_FILE" 2>&1 &

        WAVE_PIDS+=("$!")
        WAVE_SHARDS+=("$SHARD")
    done


    for i in "${!WAVE_PIDS[@]}"; do

        PID="${WAVE_PIDS[$i]}"
        SHARD="${WAVE_SHARDS[$i]}"

        echo "[INFO] Waiting for shard ${SHARD} (PID=${PID})..."

        set +e
        wait "$PID"
        EXIT_CODE=$?
        set -e

        LOG_FILE="${OUTPUT_DIR}/${PREFIX_OUT}${SHARD}${LOG_SUFFIX}"

        if [[ "$EXIT_CODE" -ne 0 ]]; then
            FAILED=$((FAILED + 1))

            echo "[ERROR] Shard ${SHARD} failed."
            echo "[ERROR] Exit code: ${EXIT_CODE}"
            echo "[ERROR] Log: ${LOG_FILE}"
        else
            COMPLETED=$((COMPLETED + 1))

            echo "[OK] Shard ${SHARD} completed."
        fi
    done

    unset WAVE_PIDS
    unset WAVE_SHARDS

done


echo
echo "======================================"
echo "[INFO] SUM R-Align inference finished."
echo "[INFO] Completed shards: ${COMPLETED}"
echo "[INFO] Failed shards   : ${FAILED}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "======================================"


if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi

exit 0