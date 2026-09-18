#!/bin/bash

# ======================================================
# User configuration
# ======================================================

CORE_NAME="${CORE_NAME:-qwen-sft}"

# Paths can be overridden through environment variables.
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-sft/qwen-sft/qwen-sft.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints}"
LFACTORY_DIR="${LFACTORY_DIR:-LLaMA-Factory}"

# Training parameters.
EPOCHS="${EPOCHS:-10}"
LORA_RANK="${LORA_RANK:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"


RESUME_DIR="${RESUME_DIR:-}"
RESUME_CKPT="${RESUME_CKPT:-}"

# If RESUME_DIR is provided but RESUME_CKPT is empty,
# automatically select the latest checkpoint-* folder.
AUTO_PICK_LATEST_CKPT="${AUTO_PICK_LATEST_CKPT:-1}"


# ======================================================
# Automatic nohup detach
# ======================================================

if [[ -z "${NOHUP_WRAPPED:-}" ]]; then

    export NOHUP_WRAPPED=1

    OUTER_LOG="/tmp/$(basename "$0").nohup.log"

    nohup "$0" "$@" \
        > "$OUTER_LOG" 2>&1 &

    echo "[INFO] Script detached with nohup."
    exit 0

fi


set -euo pipefail


# ======================================================
# Run name
# ======================================================

DATE_TAG="$(date '+%d%m%y')"
TIME_TAG="$(date '+%H%M%S')"

LR_TAG="$(
    printf "%.0e" "$LEARNING_RATE" |
    sed -E \
        's/e-0([0-9]+)/e-\1/; s/e\+0([0-9]+)/e+\1/'
)"

RUN_NAME=""
RUN_DIR=""
LATEST_CKPT=""


# ======================================================
# Resume configuration
# ======================================================

if [[ -n "$RESUME_CKPT" ]]; then
    RESUME_DIR="$(dirname "$RESUME_CKPT")"
fi


if [[ -n "$RESUME_DIR" ]]; then

    # Resume mode: reuse the existing run directory.
    RUN_DIR="$RESUME_DIR"
    RUN_NAME="$(basename "$RUN_DIR")"

    if [[ -z "$RESUME_CKPT" && "$AUTO_PICK_LATEST_CKPT" -eq 1 ]]; then

        if compgen -G "$RUN_DIR/checkpoint-*" > "/dev/null"; then

            LATEST_CKPT="$(
                find "$RUN_DIR" \
                    -maxdepth 1 \
                    -type d \
                    -name "checkpoint-*" \
                | awk -F'-' '{print $NF " " $0}' \
                | sort -n \
                | tail -1 \
                | cut -d' ' -f2-
            )"

        fi

    else

        LATEST_CKPT="$RESUME_CKPT"

    fi


    if [[ -z "$LATEST_CKPT" ]]; then
        echo "[ERROR] Resume requested but no checkpoint was found."
        exit 1
    fi


    if [[ ! -d "$LATEST_CKPT" ]]; then
        echo "[ERROR] Resume checkpoint does not exist."
        exit 1
    fi

else

    # Fresh training mode.
    RUN_NAME="${CORE_NAME}_${EPOCHS}e_rank${LORA_RANK}_lr${LR_TAG}_${DATE_TAG}_${TIME_TAG}"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

fi


# ======================================================
# Run files
# ======================================================

LOG_FILE="${RUN_DIR}/train.log"
CONFIG_YAML="${RUN_DIR}/config.patched.yaml"

mkdir -p "$RUN_DIR"


# ======================================================
# Validate configuration
# ======================================================

if [[ ! -f "$CONFIG_TEMPLATE" ]]; then
    echo "[ERROR] Configuration template was not found."
    exit 1
fi


if [[ ! -d "$LFACTORY_DIR" ]]; then
    echo "[ERROR] LLaMA-Factory directory was not found."
    exit 1
fi


# ======================================================
# Generate patched YAML
# ======================================================

CONFIG_TEMPLATE="$CONFIG_TEMPLATE" \
CONFIG_YAML="$CONFIG_YAML" \
RUN_DIR="$RUN_DIR" \
EPOCHS="$EPOCHS" \
LORA_RANK="$LORA_RANK" \
LEARNING_RATE="$LEARNING_RATE" \
PER_DEVICE_TRAIN_BATCH_SIZE="$PER_DEVICE_TRAIN_BATCH_SIZE" \
GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
LATEST_CKPT="$LATEST_CKPT" \
python3 - <<'PY'
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    print(
        "[ERROR] PyYAML is not installed. "
        "Install it with: pip install pyyaml",
        file=sys.stderr,
    )
    raise


template = Path(os.environ["CONFIG_TEMPLATE"])
output = Path(os.environ["CONFIG_YAML"])

data = yaml.safe_load(
    template.read_text(encoding="utf-8")
)


# ------------------------------------------------------
# Training parameters
# ------------------------------------------------------

data["output_dir"] = os.environ["RUN_DIR"]

data["num_train_epochs"] = int(
    os.environ["EPOCHS"]
)

lora_rank = int(
    os.environ["LORA_RANK"]
)

data["lora_rank"] = lora_rank
data["lora_alpha"] = lora_rank * 2

data["learning_rate"] = float(
    os.environ["LEARNING_RATE"]
)

data["per_device_train_batch_size"] = int(
    os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"]
)

data["gradient_accumulation_steps"] = int(
    os.environ["GRADIENT_ACCUMULATION_STEPS"]
)


# ------------------------------------------------------
# Output / resume settings
# ------------------------------------------------------

data["overwrite_output_dir"] = False

# Preserve optimizer, scheduler, and trainer state.
data["save_only_model"] = False

if "overwrite_cache" in data:
    data["overwrite_cache"] = False


latest_ckpt = os.environ.get(
    "LATEST_CKPT",
    "",
).strip()

if latest_ckpt:
    data["resume_from_checkpoint"] = latest_ckpt
else:
    data.pop(
        "resume_from_checkpoint",
        None,
    )


output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.write_text(
    yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


if latest_ckpt:
    print("[INFO] Configuration prepared for resume training.")
else:
    print("[INFO] Configuration prepared for fresh training.")

PY


# ======================================================
# Training information
# ======================================================

echo "[INFO] RUN_NAME      : $RUN_NAME"
echo "[INFO] EPOCHS        : $EPOCHS"
echo "[INFO] LORA_RANK     : $LORA_RANK"
echo "[INFO] LEARNING_RATE : $LEARNING_RATE"
echo "[INFO] BATCH_SIZE    : $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "[INFO] GRAD_ACCUM    : $GRADIENT_ACCUMULATION_STEPS"

if [[ -n "$LATEST_CKPT" ]]; then
    echo "[INFO] MODE          : RESUME"
else
    echo "[INFO] MODE          : FRESH"
fi

echo "--------------------------------------------------"


# ======================================================
# Main training command
# ======================================================

ORIG_DIR="$(pwd)"

cd "$LFACTORY_DIR"


set +e

env \
    PILLOW_TRUNCATED_IMAGES="1" \
    PYTHONWARNINGS="ignore:PIL.Image.DecompressionBombWarning" \
    llamafactory-cli train "$CONFIG_YAML" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

set -e


cd "$ORIG_DIR"


# ======================================================
# Finish
# ======================================================

if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "=== TRAINING COMPLETED SUCCESSFULLY ==="
else
    echo "=== TRAINING FAILED (exit code: $EXIT_CODE) ==="
fi

exit "$EXIT_CODE"