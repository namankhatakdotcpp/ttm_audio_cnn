#!/usr/bin/env bash
# run_ablations.sh — run all 4 ablation experiments sequentially
#
# Each experiment overrides the base train.yaml with a configs/experiment/*.yaml
# and gets a unique WandB run name via the experiment_name field in each yaml.
#
# Usage:
#   chmod +x scripts/run_ablations.sh
#   ./scripts/run_ablations.sh
#
# Optional: set ANNOTATION_PATH and VIDEO_DIR env vars to avoid editing configs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default paths — override via environment variables
ANNOTATION_PATH="${ANNOTATION_PATH:-${HOME}/ego4d_data/v2/annotations/av_train.json}"
VIDEO_DIR="${VIDEO_DIR:-${HOME}/ego4d_data/v2/full_scale}"

# Array of (experiment_yaml_name, human_label) pairs
ABLATIONS=(
    "ablation_no_mel:No Mel filterbank (linear STFT)"
    "ablation_global_norm:Global normalization instead of per-clip"
    "ablation_7x7_conv:Standard 7x7 conv (vs 7x1)"
    "ablation_no_augment:No augmentation pipeline"
)

echo "============================================================"
echo " TTM Audio CNN — Ablation Suite"
echo " $(date)"
echo "============================================================"
echo ""

FAILED=()

for entry in "${ABLATIONS[@]}"; do
    NAME="${entry%%:*}"
    LABEL="${entry##*:}"

    echo "------------------------------------------------------------"
    echo " Starting ablation: ${LABEL}"
    echo " Config:            configs/experiment/${NAME}.yaml"
    echo " Time:              $(date)"
    echo "------------------------------------------------------------"

    python "${SCRIPT_DIR}/train_audio_cnn.py" \
        "+experiment=${NAME}" \
        "annotation_path=${ANNOTATION_PATH}" \
        "video_dir=${VIDEO_DIR}" \
        "hydra.run.dir=outputs/ablations/${NAME}" \
        || {
            echo "[WARN] Ablation '${NAME}' failed — continuing with next."
            FAILED+=("${NAME}")
        }

    echo ""
    echo " Finished ablation: ${LABEL}"
    echo ""
done

echo "============================================================"
echo " All ablations attempted."
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo " FAILED runs: ${FAILED[*]}"
    exit 1
else
    echo " All runs completed successfully."
fi
echo " $(date)"
echo "============================================================"
