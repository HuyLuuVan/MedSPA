#!/bin/bash
set -euo pipefail


# ======================================================
# Shard job settings
# ======================================================

RUN_NAME="${RUN_NAME:-CheXbert_MIMIC_Labeling}"

INPUT_DIR="${INPUT_DIR:-src/r1-vl/data/shard}"
OUTPUT_DIR="${OUTPUT_DIR:-src/r1-vl/data/label}"

PREFIX="${PREFIX:-qwen_shard}"
TOTAL_SHARDS="${TOTAL_SHARDS:-128}"


# ======================================================
# Script and GPU configuration
# ======================================================

SCRIPT="${SCRIPT:-src/r1-vl/data_distillation/reasoning_eval/chestXbert/reasoning_label.py}"

NUM_GPUS="${NUM_GPUS:-8}"

BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_EVERY="${SAVE_EVERY:-1000}"

MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-30}"


mkdir -p "$OUTPUT_DIR"


# ======================================================
# Run information
# ======================================================

echo "=================================================="
echo "[CheXbert] RUN_NAME     : $RUN_NAME"
echo "[CheXbert] TOTAL_SHARDS : $TOTAL_SHARDS"
echo "[CheXbert] NUM_GPUS     : $NUM_GPUS"
echo "[CheXbert] BATCH_SIZE   : $BATCH_SIZE"
echo "[CheXbert] SAVE_EVERY   : $SAVE_EVERY"
echo "[CheXbert] MAX_RETRIES  : $MAX_RETRIES"
echo "[CheXbert] RETRY_DELAY  : $RETRY_DELAY"
echo "=================================================="


declare -a GPU_PIDS


# ======================================================
# Main loop
# ======================================================

for ((i=0; i<TOTAL_SHARDS; i++)); do

    GPU_ID=$((i % NUM_GPUS))
    BASE="${PREFIX}${i}"

    INPUT_FILE="${INPUT_DIR}/${BASE}.json"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE}_label.json"
    LOG_FILE="${OUTPUT_DIR}/${BASE}_label.log"


    # Wait for the previous job on the same GPU.
    if [ -n "${GPU_PIDS[$GPU_ID]:-}" ]; then

        if kill -0 "${GPU_PIDS[$GPU_ID]}" 2>/dev/null; then

            echo "[GPU $GPU_ID] Waiting for previous job..."

            wait "${GPU_PIDS[$GPU_ID]}"

        fi

    fi


    echo "[GPU $GPU_ID] Launching shard $i"


    CMD="
attempt=1

while [ \$attempt -le $MAX_RETRIES ]; do

    echo \"[GPU $GPU_ID][shard $i] Attempt \$attempt\"

    CUDA_VISIBLE_DEVICES=$GPU_ID python \"$SCRIPT\" \
        --input \"$INPUT_FILE\" \
        --output \"$OUTPUT_FILE\" \
        --batch_size \"$BATCH_SIZE\" \
        --save_every \"$SAVE_EVERY\"

    ec=\$?

    if [ \$ec -eq 0 ]; then
        echo \"[GPU $GPU_ID][shard $i] Success\"
        exit 0
    fi

    echo \"[GPU $GPU_ID][shard $i] Failed with code \$ec\"

    if [ \$attempt -lt $MAX_RETRIES ]; then
        echo \"[GPU $GPU_ID][shard $i] Retrying in $RETRY_DELAY seconds...\"
        sleep \"$RETRY_DELAY\"
    fi

    attempt=\$((attempt + 1))

done

echo \"[GPU $GPU_ID][shard $i] Failed after $MAX_RETRIES attempts.\"

exit 1
"


    nohup bash -c "$CMD" \
        >> "$LOG_FILE" 2>&1 &

    GPU_PIDS[$GPU_ID]=$!

done

for pid in "${GPU_PIDS[@]}"; do

    if [ -n "${pid:-}" ]; then

        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
        fi

    fi

done


echo "=== ALL SHARDS COMPLETED ==="