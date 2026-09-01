#!/usr/bin/env bash
#
# Train EfficientDet on the custom 3D-print anomaly dataset.
#
#   ./run_train.sh                 # from scratch on the anomaly data (default)
#   ./run_train.sh anomaly         # same, explicit
#   ./run_train.sh coco            # stage 1: pretrain on MS COCO 2017
#   ./run_train.sh transfer        # stage 2: anomaly, warm-started from the COCO run
#   ./run_train.sh all             # coco, then transfer
#
# Every knob below can be overridden from the environment, e.g.
#   EPOCHS=100 BATCH_SIZE=32 ./run_train.sh
#   FREEZE_BACKBONE=1 ./run_train.sh anomaly   # train only BiFPN+heads (150k params)
#
# GPU count is detected automatically: 1 GPU runs `python`, more runs `torchrun`
# with the learning rate scaled by the number of ranks.

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-anomaly}"

# ---- Tunables --------------------------------------------------------------
PHI="${PHI:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"          # PER GPU
WORKERS="${WORKERS:-8}"                 # PER GPU
PATIENCE="${PATIENCE:-10}"
BASE_LR="${BASE_LR:-1e-4}"              # for from-scratch runs
FINETUNE_LR="${FINETUNE_LR:-5e-5}"      # for --init-from runs
FREEZE_BACKBONE="${FREEZE_BACKBONE:-}"  # 1/0; unset = mode default (see below)
RUN_EVAL="${RUN_EVAL:-1}"               # 1 = evaluate on the test split when done
DRY_RUN="${DRY_RUN:-0}"                 # 1 = print the resolved commands, run nothing

ANOMALY_DIR="${ANOMALY_DIR:-data/anomaly}"
COCO_DIR="${COCO_DIR:-coco}"
COCO_CKPT="${COCO_CKPT:-checkpoints_coco}"
ANOMALY_CKPT="${ANOMALY_CKPT:-checkpoints_anomaly}"

# ---- Backbone freezing -----------------------------------------------------
# The backbone is 96% of the model (3.70M of 3.85M params at phi 0) and is
# already ImageNet-pretrained by timm. Freezing it drops trainable params to
# 150k, which is the main lever against overfitting on ~1.9k images.
#
# Default on for transfer (the standard fine-tuning recipe, and the pretrained
# features are the whole point of --init-from) and off for a direct anomaly run
# so the plain command stays a full fine-tune. FREEZE_BACKBONE=1/0 overrides.
if [ -z "$FREEZE_BACKBONE" ]; then
    case "$MODE" in
        transfer|all) FREEZE_BACKBONE=1 ;;
        *)            FREEZE_BACKBONE=0 ;;
    esac
fi

# ---- GPU selection ---------------------------------------------------------
# This box is 4x RTX A5000 on plain PCIe (no NVLink), so NCCL peer-to-peer is
# usually blocked by IOMMU/ACS and every collective hangs. Route them through
# host memory by default; set NCCL_P2P_DISABLE=0 if you have verified P2P works.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED=$(nvidia-smi -L | wc -l)
else
    DETECTED=0
fi

# The anomaly set is ~1.9k images: one epoch is ~116 steps at batch 16, so DDP
# saves minutes while adding a real hang risk. COCO is 118k images, where it
# saves hours. Default accordingly; NGPU=<n> overrides either way.
if [ -n "${NGPU:-}" ]; then
    :                                   # explicit override wins
elif [ "$MODE" = "coco" ] || [ "$MODE" = "all" ]; then
    NGPU="$DETECTED"
else
    NGPU=1
fi
[ "$NGPU" -gt "$DETECTED" ] && NGPU="$DETECTED"

if [ "$NGPU" -gt 1 ]; then
    LAUNCH=(torchrun --nproc_per_node="$NGPU")
    # Effective batch is BATCH_SIZE x NGPU, so scale the LR to match.
    BASE_LR=$(python3 -c "print(f'{$BASE_LR * $NGPU:.2e}')")
    FINETUNE_LR=$(python3 -c "print(f'{$FINETUNE_LR * $NGPU:.2e}')")
else
    LAUNCH=(python3)
fi

echo "=========================================================="
echo " mode          : $MODE"
echo " GPUs          : $NGPU of $DETECTED  ($([ "$NGPU" -gt 1 ] && echo torchrun || echo single-process))"
echo " NCCL_P2P      : disabled=$NCCL_P2P_DISABLE"
echo " phi           : $PHI"
echo " batch/GPU     : $BATCH_SIZE   (effective $((BATCH_SIZE * (NGPU > 0 ? NGPU : 1))))"
echo " lr            : scratch=$BASE_LR  finetune=$FINETUNE_LR"
echo " freeze bbone  : $FREEZE_BACKBONE   (trainable: $([ "$FREEZE_BACKBONE" = "1" ] && echo "150k BiFPN+heads" || echo "3.85M all"))"
echo " epochs        : $EPOCHS"
echo "=========================================================="

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '  [dry-run] '; printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

# ---- Preflight: fail before burning GPU time, not 20 minutes in ------------
require_dir()  { [ -d "$1" ] || { echo "MISSING directory: $1" >&2; exit 1; }; }
require_file() { [ -f "$1" ] || { echo "MISSING file: $1"      >&2; exit 1; }; }

check_anomaly_data() {
    require_dir  "$ANOMALY_DIR/images"
    for s in train val test; do
        require_file "$ANOMALY_DIR/annotations/instances_${s}.json"
    done
    echo "Anomaly data OK: $(ls "$ANOMALY_DIR/images" | wc -l) images"
}

# download_coco.py extracts to <root>/images/train2017, but a manually
# assembled tree often has <root>/train2017 directly. Support both.
COCO_TRAIN_DIR=""
COCO_VAL_DIR=""
check_coco_data() {
    if   [ -d "$COCO_DIR/images/train2017" ]; then
        COCO_TRAIN_DIR="$COCO_DIR/images/train2017"
        COCO_VAL_DIR="$COCO_DIR/images/val2017"
    elif [ -d "$COCO_DIR/train2017" ]; then
        COCO_TRAIN_DIR="$COCO_DIR/train2017"
        COCO_VAL_DIR="$COCO_DIR/val2017"
    else
        echo "MISSING: no train2017 under $COCO_DIR/images/ or $COCO_DIR/" >&2
        echo "         set COCO_DIR=<root> to point at your COCO tree" >&2
        exit 1
    fi
    require_dir  "$COCO_VAL_DIR"
    require_file "$COCO_DIR/annotations/instances_train2017.json"
    require_file "$COCO_DIR/annotations/instances_val2017.json"
    echo "COCO data OK: $COCO_TRAIN_DIR"
}

# ---- Training stages -------------------------------------------------------
train_coco() {
    check_coco_data
    echo ">>> Stage: COCO pretraining -> $COCO_CKPT"
    run "${LAUNCH[@]}" train.py \
        --train-images "$COCO_TRAIN_DIR" \
        --train-ann    "$COCO_DIR/annotations/instances_train2017.json" \
        --val-images   "$COCO_VAL_DIR" \
        --val-ann      "$COCO_DIR/annotations/instances_val2017.json" \
        --phi "$PHI" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$BASE_LR" \
        --workers "$WORKERS" \
        --patience "$PATIENCE" \
        --checkpoint-dir "$COCO_CKPT"
}

train_anomaly() {
    check_anomaly_data
    local extra=()
    if [ "${1:-scratch}" = "transfer" ]; then
        require_file "$COCO_CKPT/best.pth"
        extra+=(--init-from "$COCO_CKPT/best.pth" --lr "$FINETUNE_LR")
        echo ">>> Stage: anomaly fine-tune from $COCO_CKPT/best.pth -> $ANOMALY_CKPT"
    else
        extra+=(--lr "$BASE_LR")
        echo ">>> Stage: anomaly from scratch -> $ANOMALY_CKPT"
    fi
    # Applies to both paths: freezing is just as useful without --init-from,
    # since the backbone carries ImageNet weights on every run.
    if [ "$FREEZE_BACKBONE" = "1" ]; then
        extra+=(--freeze-backbone)
    fi

    # --keep-empty keeps the ~50% "Normal" images as negatives; without it they
    # are silently dropped. --test-fraction 0 disables the random carve-out,
    # since this dataset ships its own instances_test.json.
    run "${LAUNCH[@]}" train.py \
        --train-images "$ANOMALY_DIR/images" \
        --train-ann    "$ANOMALY_DIR/annotations/instances_train.json" \
        --val-images   "$ANOMALY_DIR/images" \
        --val-ann      "$ANOMALY_DIR/annotations/instances_val.json" \
        --phi "$PHI" \
        --test-fraction 0 \
        --keep-empty \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --patience "$PATIENCE" \
        --checkpoint-dir "$ANOMALY_CKPT" \
        "${extra[@]}"
}

eval_anomaly() {
    [ "$RUN_EVAL" = "1" ] || return 0
    echo ""
    echo ">>> Evaluating $ANOMALY_CKPT/best.pth on the anomaly test split"
    # --split all because instances_test.json IS the test split already;
    # --split test would re-carve a 5% slice out of it.
    run python3 evaluate.py \
        --images "$ANOMALY_DIR/images" \
        --ann    "$ANOMALY_DIR/annotations/instances_test.json" \
        --split all \
        --keep-empty \
        --phi "$PHI" \
        --checkpoint "$ANOMALY_CKPT/best.pth" \
        --workers "$WORKERS"
}

case "$MODE" in
    anomaly)  train_anomaly scratch  ; eval_anomaly ;;
    coco)     train_coco ;;
    transfer) train_anomaly transfer ; eval_anomaly ;;
    all)      train_coco ; train_anomaly transfer ; eval_anomaly ;;
    *) echo "Unknown mode: $MODE (expected anomaly|coco|transfer|all)" >&2; exit 1 ;;
esac

echo ""
echo "Done. Checkpoints and loss curve in: $ANOMALY_CKPT"
