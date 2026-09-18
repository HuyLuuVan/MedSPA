#!/bin/bash
set -euo pipefail

PYTHON_SCRIPT="${PYTHON_SCRIPT:-src/r1-vl/data_distillation/mimic/qwen3vl_reasoning.py}"

OUT_DIR="${OUT_DIR:-src/r1-vl/data}"
SHARD_DIR="${SHARD_DIR:-${OUT_DIR}/shard}"

mkdir -p "$SHARD_DIR"

if [ -n "${SERVERS_CSV:-}" ]; then
    IFS=',' read -r -a SERVERS <<< "$SERVERS_CSV"
else
    SERVERS=(
        "http://localhost:8000/v1"
        "http://localhost:8001/v1"
        "http://localhost:8002/v1"
        "http://localhost:8003/v1"
        "http://localhost:8004/v1"
        "http://localhost:8005/v1"
        "http://localhost:8006/v1"
        "http://localhost:8007/v1"
    )
fi

NUM_SHARDS="${NUM_SHARDS:-128}"

IMAGES_INDEX="${IMAGES_INDEX:-../mimic_cxr_all_images.txt}"
MIMIC_FINDINGS_JSON="${MIMIC_FINDINGS_JSON:-../mimic_gt_findings.json}"
MIMIC_ANNOT_JSON="${MIMIC_ANNOT_JSON:-../mimic_annotation.json}"

MAX_STEPS="${MAX_STEPS:-10}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-0}"

ONLY_LIST="${ONLY_LIST-src/r1-vl/data/mimic/revised/bad_samples.txt}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"

if [ "${#SERVERS[@]}" -eq 0 ]; then
    echo "[ERROR] No servers configured."
    exit 1
fi

if [ "$#" -eq 0 ]; then
    SHARDS_TO_RUN=()

    for ((i=0; i<NUM_SHARDS; i++)); do
        SHARDS_TO_RUN+=("$i")
    done
else
    SHARDS_TO_RUN=("$@")
fi

for SHARD_ID in "${SHARDS_TO_RUN[@]}"; do
    if ! [[ "$SHARD_ID" =~ ^[0-9]+$ ]] \
        || [ "$SHARD_ID" -ge "$NUM_SHARDS" ]; then
        echo "[ERROR] Invalid shard ID: $SHARD_ID"
        exit 1
    fi

    SERVER_INDEX=$((SHARD_ID % ${#SERVERS[@]}))
    SERVER="${SERVERS[$SERVER_INDEX]}"

    OUT_JSON="${SHARD_DIR}/qwen_shard${SHARD_ID}.json"
    LOG_FILE="${SHARD_DIR}/qwen_shard${SHARD_ID}.log"

    ONLY_LIST_ARGS=()

    if [ -n "$ONLY_LIST" ]; then
        ONLY_LIST_ARGS=(
            --only_list "$ONLY_LIST"
        )
    fi

    echo \
        "[SHARD $SHARD_ID/$NUM_SHARDS] " \
        "Launching on server $SERVER_INDEX."

    QWEN_API_BASE="$SERVER" \
    QWEN_API_KEY="$QWEN_API_KEY" \
    QWEN_MODEL="$QWEN_MODEL" \
    nohup python "$PYTHON_SCRIPT" \
        --images_index "$IMAGES_INDEX" \
        --mimic_findings_json "$MIMIC_FINDINGS_JSON" \
        --mimic_annotation_json "$MIMIC_ANNOT_JSON" \
        --output_json "$OUT_JSON" \
        --split "$SPLIT" \
        --max_steps "$MAX_STEPS" \
        --save_every "$SAVE_EVERY" \
        --num_shards "$NUM_SHARDS" \
        --shard_id "$SHARD_ID" \
        --limit "$LIMIT" \
        "${ONLY_LIST_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    sleep 1
done

echo "[DONE] Requested distillation shards launched."