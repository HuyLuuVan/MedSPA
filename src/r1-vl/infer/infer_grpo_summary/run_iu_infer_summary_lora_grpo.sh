#!/bin/bash

# ======================================================
#   AUTO NOHUP DETACH (only once)
# ======================================================
if [[ -z "${NOHUP_WRAPPED:-}" ]]; then
    export NOHUP_WRAPPED=1
    OUTER_LOG="/tmp/$(basename "$0").nohup.log"
    nohup "$0" "$@" > "$OUTER_LOG" 2>&1 &
    echo "[INFO] Script detached with nohup (PID=$!)."
    echo "[INFO] Outer log: $OUTER_LOG"
    exit 0
fi

set -u

# ======================================================
#   Config
# ======================================================

# Base + adapters:
#  - SFT adapter will be merged into base
#  - GRPO adapter will be active for inference
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-/mimlab/luu/R1-VL-Med/checkpoints/mimic/sft/summary/qwen3vl_8b_lora-sft-s-miss28k_10e_rank16_lr1e-4_280123_154501}"
GRPO_ADAPTER_PATH="${GRPO_ADAPTER_PATH:-/mimlab/luu/R1-VL-Med/checkpoints/mimic/grpo/summary/qwen3vl_8b_grpo_summary_labelreward_1e_rank16_010226_032431}"

# I/O (mimic -> iu)
INPUT_SHARD_DIR="${INPUT_SHARD_DIR:-/mimlab/luu/R1-VL-Med/results/iu/grpo/qwen3vl_8b_grpo_reasoning_labelreward_lc_pen_1e_050226_045859/label}"
OUTPUT_DIR="${OUTPUT_DIR:-/mimlab/luu/R1-VL-Med/results/iu/grpo/summary/qwen3vl_8b_grpo_summary_labelreward_1e_rank16_010226_032431/qwen3vl_8b_grpo_reasoning_labelreward_lc_pen_1e_050226_045859/caption}"

PREFIX_IN="${PREFIX_IN:-test_reasoning_shard}"
SUFFIX_IN="${SUFFIX_IN:-_label.json}"

PREFIX_OUT="${PREFIX_OUT:-test_caption_shard}"
SUFFIX_OUT="${SUFFIX_OUT:-.json}"
LOG_SUFFIX="${LOG_SUFFIX:-.log}"

# Python script for GRPO summary inference (HF native)
PY_SCRIPT="${PY_SCRIPT:-src/r1-vl/infer/infer_grpo_summary/infer_summary_lora_grpo.py}"

# Image index (mimic -> iu)
ALL_IMAGES_PATH="${ALL_IMAGES_PATH:-dataset/iu/iu_cxr_all_images.txt}"

# Parallelism
NUM_GPUS="${NUM_GPUS:-8}"
NUM_SHARDS="${NUM_SHARDS:-16}"

# Generation params
TEMP="${TEMP:-0.2}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW="${MAX_NEW:-512}"
NUM_BEAMS="${NUM_BEAMS:-1}"

# Image preprocess / attention
IMG_SIZE="${IMG_SIZE:-1024}"
PAD_COLOR="${PAD_COLOR:-0}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"  # sdpa | flash_attention_2 | eager

# Saving / resume
SAVE_EVERY="${SAVE_EVERY:-50}"
RESUME="${RESUME:-1}"          # 1 = resume, 0 = no resume

# Optional: allow custom Python binary
PYTHON_BIN="${PYTHON_BIN:-python}"

# ======================================================
#   Setup (after config)
# ======================================================
mkdir -p "$OUTPUT_DIR"

# Load Telegram helper functions (send_telegram, load_telegram_env, etc.)
source notification/telegram_utils.sh

# Load BOT_TOKEN and CHAT_ID from ~/.telegram_env (if available)
load_telegram_env

RUN_NAME="${RUN_NAME:-Qwen3VL_GRPO_Summary_Infer_IU_Test}"
START_TIME="$(date '+%Y-%m-%d %H:%M')"
HOSTNAME="$(hostname)"

echo "[INFO] Starting: $RUN_NAME at $START_TIME on $HOSTNAME"

# --------------------------------------------
#  30-min “still running” notification
# --------------------------------------------
PARENT_PID=$$
notify_if_still_running_after \
  1800 \
  "$PARENT_PID" \
  "⏳ Run \"$RUN_NAME\" is still running on $HOSTNAME
Started at: $START_TIME
Elapsed: 30 minutes."

echo "[INFO] PY_SCRIPT          : $PY_SCRIPT"
echo "[INFO] BASE_MODEL         : $BASE_MODEL"
echo "[INFO] SFT_ADAPTER_PATH   : $SFT_ADAPTER_PATH"
echo "[INFO] GRPO_ADAPTER_PATH  : $GRPO_ADAPTER_PATH"
echo "[INFO] INPUT_SHARD_DIR    : $INPUT_SHARD_DIR"
echo "[INFO] OUTPUT_DIR         : $OUTPUT_DIR"
echo "[INFO] NUM_GPUS/SHARDS    : $NUM_GPUS / $NUM_SHARDS"
echo "[INFO] ATTN/IMG_SIZE      : $ATTN_IMPL / $IMG_SIZE"

# Build resume flag
RESUME_FLAG=""
if [ "$RESUME" -eq 1 ]; then
  RESUME_FLAG="--resume"
  echo "[INFO] Resume enabled."
else
  echo "[INFO] Resume disabled."
fi

# ======================================================
#   Launch shards (balanced across GPUs) with nohup
# ======================================================
declare -a GPU_PIDS
FAILED=0

echo "[INFO] Launching shards ..."

for SHARD in $(seq 0 $((NUM_SHARDS-1))); do
  GPU_ID=$(( SHARD % NUM_GPUS ))

  BASE_IN="${PREFIX_IN}${SHARD}"
  BASE_OUT="${PREFIX_OUT}${SHARD}"

  IN_JSON="${INPUT_SHARD_DIR}/${BASE_IN}${SUFFIX_IN}"
  OUT_JSON="${OUTPUT_DIR}/${BASE_OUT}${SUFFIX_OUT}"
  LOG="${OUTPUT_DIR}/${BASE_OUT}${LOG_SUFFIX}"

  if [ ! -f "$IN_JSON" ]; then
    echo "[WARN] Missing input shard file: $IN_JSON"
    continue
  fi

  echo "[GPU $GPU_ID] Launching shard $SHARD"
  echo "  Input : $IN_JSON"
  echo "  Output: $OUT_JSON"
  echo "  Log   : $LOG"

  CUDA_VISIBLE_DEVICES=$GPU_ID nohup "$PYTHON_BIN" "$PY_SCRIPT" \
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
    $RESUME_FLAG \
    > "$LOG" 2>&1 &

  GPU_PIDS[$SHARD]=$!
done

# ======================================================
#   Wait for all shards and detect failures
# ======================================================
for SHARD in $(seq 0 $((NUM_SHARDS-1))); do
  PID="${GPU_PIDS[$SHARD]:-}"

  # Skip shards not launched
  if [ -z "$PID" ]; then
    continue
  fi

  if kill -0 "$PID" 2>/dev/null; then
    echo "[INFO] Waiting for shard $SHARD (PID $PID)..."
    wait "$PID"
    EXIT_CODE=$?

    LOG_FILE="${OUTPUT_DIR}/${PREFIX_OUT}${SHARD}${LOG_SUFFIX}"
    if [ $EXIT_CODE -ne 0 ]; then
      FAILED=$((FAILED+1))
      echo "[ERROR] Shard $SHARD failed with exit code $EXIT_CODE (see $LOG_FILE)"

      if has_telegram_config; then
        send_telegram "❌ Shard $SHARD FAILED in \"$RUN_NAME\" (exit=$EXIT_CODE)
Host: $HOSTNAME
Log: $LOG_FILE"
      fi
    else
      echo "[OK] Shard $SHARD finished successfully."
    fi
  fi
done

END_TIME="$(date '+%Y-%m-%d %H:%M')"

# ======================================================
#   Final Telegram notification
# ======================================================
if has_telegram_config; then
  if [ "$FAILED" -eq 0 ]; then
    send_telegram "✅ Run \"$RUN_NAME\" completed successfully
🧩 Shards: $NUM_SHARDS
🖥️ GPUs: $NUM_GPUS
🕒 Start: $START_TIME
🏁 End: $END_TIME
📂 Output: $OUTPUT_DIR
Host: $HOSTNAME"
  else
    send_telegram "⚠️ Run \"$RUN_NAME\" finished with $FAILED failed shard(s)
🧩 Shards: $NUM_SHARDS
🖥️ GPUs: $NUM_GPUS
🕒 Start: $START_TIME
🏁 End: $END_TIME
📂 Output: $OUTPUT_DIR
Host: $HOSTNAME"
  fi
fi

echo "=== ALL SHARDS COMPLETE (failed=$FAILED) ==="
exit 0