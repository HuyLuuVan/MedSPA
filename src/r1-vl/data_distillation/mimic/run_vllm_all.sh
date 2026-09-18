#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8000}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

LOG_DIR="${LOG_DIR:-src/r1-vl/data/mimic/log}"

CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache}"
TMPDIR="${TMPDIR:-${HOME}/tmp}"

export TMPDIR
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export XDG_CACHE_HOME="$CACHE_ROOT"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"

mkdir -p \
    "$TMPDIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" \
    "$LOG_DIR"

VLLM_BIN="$(command -v vllm || true)"

if [ -z "$VLLM_BIN" ]; then
    echo "[ERROR] vllm was not found in the active environment."
    exit 1
fi

if [ "$#" -eq 0 ]; then
    GPUS=(0 1 2 3 4 5 6 7)
else
    GPUS=("$@")
fi

for GPU in "${GPUS[@]}"; do
    if ! [[ "$GPU" =~ ^[0-9]+$ ]] || [ "$GPU" -gt 7 ]; then
        echo "[ERROR] Invalid GPU ID: $GPU"
        exit 1
    fi

    PORT=$((BASE_PORT + GPU))
    LOG_FILE="${LOG_DIR}/vllm${GPU}.log"

    OLD_PIDS="$(
        pgrep -f "vllm serve .*--port ${PORT}" \
        || true
    )"

    if [ -n "$OLD_PIDS" ]; then
        echo "[GPU $GPU] Stopping existing server on port $PORT."

        for PID in $OLD_PIDS; do
            kill "$PID" 2>/dev/null || true
        done

        sleep 2

        for PID in $OLD_PIDS; do
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null || true
            fi
        done
    fi

    echo "[GPU $GPU] Starting vLLM on port $PORT."

    CUDA_VISIBLE_DEVICES="$GPU" \
    nohup "$VLLM_BIN" serve "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --enforce-eager \
        > "$LOG_FILE" 2>&1 &

    sleep 1
done

echo "[DONE] Selected vLLM servers started."