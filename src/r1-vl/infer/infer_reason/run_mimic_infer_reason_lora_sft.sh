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

PY_SCRIPT="${PY_SCRIPT:-${REPO_ROOT}/src/r1-vl/infer/infer_reason/infer_sft_reason.py}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
LORA_PATH="${LORA_PATH:-${REPO_ROOT}/checkpoints/sft_reason}"

MIMIC_SHARD_DIR="${MIMIC_SHARD_DIR:-${REPO_ROOT}/data/mimic/test_shards}"

ALL_IMAGES_PATH="${ALL_IMAGES_PATH:-${REPO_ROOT}/dataset/mimic_cxr_all_images.txt}"
FINDINGS_JSON="${FINDINGS_JSON:-${REPO_ROOT}/dataset/mimic_gt_findings.json}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/mimic/sft_reason}"

PREFIX_IN="${PREFIX_IN:-annotation__test_shard}"
SUFFIX_IN="${SUFFIX_IN:-.json}"

PREFIX_OUT="${PREFIX_OUT:-test_reasoning_shard}"
SUFFIX_OUT="${SUFFIX_OUT:-.json}"
LOG_SUFFIX="${LOG_SUFFIX:-.log}"

SPLIT="${SPLIT:-test}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_SHARDS="${NUM_SHARDS:-16}"

MAX_ROUNDS="${MAX_ROUNDS:-10}"
TEMP="${TEMP:-0.2}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW="${MAX_NEW:-512}"
NUM_BEAMS="${NUM_BEAMS:-1}"

SAVE_EVERY="${SAVE_EVERY:-50}"
RESUME="${RESUME:-1}"
PRETRAINED_ONLY="${PRETRAINED_ONLY:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"

if [[ -n "${LLAMAFACTORY_SRC:-}" ]]; then
    export LLAMAFACTORY_SRC
fi

mkdir -p "$OUTPUT_DIR"


if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "[ERROR] Inference script not found: $PY_SCRIPT"
    exit 1
fi

if [[ "$PRETRAINED_ONLY" != "1" && "$PRETRAINED_ONLY" != "true" && "$PRETRAINED_ONLY" != "TRUE" ]]; then
    if [[ ! -d "$LORA_PATH" ]]; then
        echo "[ERROR] SFT LoRA adapter not found: $LORA_PATH"
        exit 1
    fi
fi

if [[ ! -d "$MIMIC_SHARD_DIR" ]]; then
    echo "[ERROR] Shard directory not found: $MIMIC_SHARD_DIR"
    exit 1
fi

if [[ ! -f "$ALL_IMAGES_PATH" ]]; then
    echo "[ERROR] Image index not found: $ALL_IMAGES_PATH"
    exit 1
fi

if [[ ! -f "$FINDINGS_JSON" ]]; then
    echo "[ERROR] Findings file not found: $FINDINGS_JSON"
    exit 1
fi


RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
    RESUME_ARGS+=(--resume)
    echo "[INFO] Resume enabled."
else
    echo "[INFO] Resume disabled."
fi


PRETRAINED_ARGS=()
if [[ "$PRETRAINED_ONLY" == "1" || "$PRETRAINED_ONLY" == "true" || "$PRETRAINED_ONLY" == "TRUE" ]]; then
    PRETRAINED_ARGS+=(--pretrained_only)
    echo "[INFO] Pretrained-only mode enabled."
fi


echo "[INFO] Starting SFT PR inference."
echo "[INFO] Script       : $PY_SCRIPT"
echo "[INFO] Base model   : $BASE_MODEL"
echo "[INFO] LoRA adapter : $LORA_PATH"
echo "[INFO] Shard dir    : $MIMIC_SHARD_DIR"
echo "[INFO] Image index  : $ALL_IMAGES_PATH"
echo "[INFO] Findings     : $FINDINGS_JSON"
echo "[INFO] Output dir   : $OUTPUT_DIR"
echo "[INFO] Split        : $SPLIT"
echo "[INFO] GPUs         : $NUM_GPUS"
echo "[INFO] Shards       : $NUM_SHARDS"


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

        SHARD_JSON="${MIMIC_SHARD_DIR}/${BASE_IN}${SUFFIX_IN}"
        OUT_JSON="${OUTPUT_DIR}/${BASE_OUT}${SUFFIX_OUT}"
        LOG_FILE="${OUTPUT_DIR}/${BASE_OUT}${LOG_SUFFIX}"

        if [[ ! -f "$SHARD_JSON" ]]; then
            echo "[ERROR] Missing shard: $SHARD_JSON"
            FAILED=$((FAILED + 1))
            continue
        fi

        echo "[GPU ${GPU_ID}] shard=${SHARD}"
        echo "  input : ${SHARD_JSON}"
        echo "  output: ${OUT_JSON}"
        echo "  log   : ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        "$PYTHON_BIN" "$PY_SCRIPT" \
            --base_model "$BASE_MODEL" \
            --model_path "$LORA_PATH" \
            --mimic_json "$SHARD_JSON" \
            --all_images_path "$ALL_IMAGES_PATH" \
            --findings_json "$FINDINGS_JSON" \
            --split "$SPLIT" \
            --output_json "$OUT_JSON" \
            --max_rounds "$MAX_ROUNDS" \
            --temperature "$TEMP" \
            --top_p "$TOP_P" \
            --max_new_tokens "$MAX_NEW" \
            --num_beams "$NUM_BEAMS" \
            --save_every "$SAVE_EVERY" \
            "${RESUME_ARGS[@]}" \
            "${PRETRAINED_ARGS[@]}" \
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
echo "[INFO] SFT PR inference finished."
echo "[INFO] Completed shards: ${COMPLETED}"
echo "[INFO] Failed shards   : ${FAILED}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "======================================"


if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi

exit 0