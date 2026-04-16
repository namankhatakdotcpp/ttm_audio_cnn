# TTM Audio CNN — S4 & S5

Modified ResNet-18 audio encoder for the Ego4D **Talking To Me (TTM)** benchmark.
This module is the S4/S5 component of a larger multi-modal TTM pipeline; its
output embeddings are consumed by the S6/S7 video–audio fusion module.

## Architecture overview

```
Raw audio (16 kHz mono)
       │
  MelExtractor          — 512-pt FFT, 128 mel bins, per-clip z-score, ±3 clamp
       │  (1, 128, T)
  AudioCNN              — ResNet-18 with conv1=(7,1)/(2,1), layer4 removed
       │  (B, 256)
  ProjectionHead        — Linear→BN→ReLU→Dropout→Linear, L2-normalized
       │  (B, 256)
  Linear classifier     — Binary TTM prediction (training only)
       │
  Output embeddings     — (T, 256) per-clip, saved as .pt for S6/S7
```

Key design choices:
- **First conv kernel (7,1)** — preserves the mel-frequency axis; (7,7) ablation
  available via `+experiment=ablation_7x7_conv`
- **layer4 removed** — outputs 256-channel features from layer3, reduces over-fitting
  on the TTM dataset
- **Per-clip z-score normalization** — robust to heterogeneous Ego4D loudness levels
- **No time-stretch augmentation** — would break temporal alignment with face tracks

## Setup

```bash
conda env create -f environment.yaml
conda activate ttm_audio
```

Hardware: NVIDIA RTX A6000 (48 GB). Single-GPU training.

## Quick smoke test (no real data needed)

```bash
python scripts/dry_run.py
```

Runs 2 epochs with synthetic mel tensors. Exits 0 on success.

## Data preparation

1. Download Ego4D V2 with `ego4d --output_directory ~/ego4d_data --datasets full_scale annotations`
2. Pre-compute and cache all Mel spectrograms:

```bash
python scripts/preprocess_audio.py \
  annotation_path=~/ego4d_data/v2/annotations/av_train.json \
  video_dir=~/ego4d_data/v2/full_scale \
  mel_cache_dir=outputs/mel_cache
```

This is resumable: re-running skips already-cached clips. Progress and
class-balance statistics are printed at completion.

## Training

```bash
python scripts/train_audio_cnn.py \
  annotation_path=~/ego4d_data/v2/annotations/av_train.json \
  video_dir=~/ego4d_data/v2/full_scale
```

Checkpoints are saved to `outputs/checkpoints/<run_name>/`.
Embeddings are extracted automatically after training to `outputs/embeddings/`.

### Fast dev run (2 batches, no WandB)

```bash
python scripts/train_audio_cnn.py trainer.fast_dev_run=true \
  annotation_path=... video_dir=...
```

## Evaluation

```bash
python scripts/evaluate.py \
  checkpoint_path=outputs/checkpoints/<run>/epoch=XX-val_auc=0.XXXX.ckpt \
  annotation_path=... video_dir=...
```

Prints a metrics table (AUC, Accuracy, F1, Precision, Recall, optimal threshold)
for val and test splits, and saves `outputs/eval/<ckpt_stem>/metrics.json`.

## Grad-CAM visualization

```bash
python scripts/visualize_gradcam.py \
  checkpoint_path=outputs/checkpoints/<run>/best.ckpt \
  annotation_path=... video_dir=... \
  n_gradcam_samples=8
```

Saves per-clip figures and a summary grid to `outputs/gradcam/<ckpt_stem>/`.

## Ablation experiments

Four ablations are pre-configured in `configs/experiment/`:

| Experiment | Config | Tests |
|---|---|---|
| No Mel filterbank | `ablation_no_mel` | Linear STFT bins vs. mel-warped |
| Global normalization | `ablation_global_norm` | Per-clip zscore vs. fixed global stats |
| 7×7 first conv | `ablation_7x7_conv` | Frequency-axis stride vs. none |
| No augmentation | `ablation_no_augment` | Full augment pipeline vs. none |

Run all four sequentially:
```bash
./scripts/run_ablations.sh
```

Or individually:
```bash
python scripts/train_audio_cnn.py +experiment=ablation_7x7_conv \
  annotation_path=... video_dir=...
```

## Output contract for S6/S7

After training, `outputs/embeddings/` contains:

```
{clip_uid}_{face_id}.pt   — float32 tensor, shape (T, 256)
manifest.csv              — clip_uid, face_id, label, pt_path, n_windows,
                            duration_sec, split
```

T = number of 0.5-second windows (stride 0.1 s) in the clip.
Each row is an L2-normalized 256-dimensional audio embedding.

## S4/S5 Handoff Protocol

This section is for the S6/S7 BiLSTM fusion team. Run these commands in order after training completes.

### Step 1 — Pre-compute Mel cache

```bash
python scripts/preprocess_audio.py \
  annotation_path=~/ego4d_data/v2/annotations/av_train.json \
  video_dir=~/ego4d_data/v2/full_scale \
  mel_cache_dir=outputs/mel_cache
```

Resumable: re-running skips already-cached clips.

### Step 2 — Train

```bash
python scripts/train_audio_cnn.py \
  annotation_path=~/ego4d_data/v2/annotations/av_train.json \
  video_dir=~/ego4d_data/v2/full_scale
```

Best checkpoint is saved to `outputs/checkpoints/<run_name>/`.
At the end of training the exact extraction command is printed to the log.

### Step 3 — Extract embeddings

```bash
python src/evaluation/extract_embeddings.py \
  --ckpt outputs/checkpoints/<run_name>/epoch=XX-val_auc=0.XXXX.ckpt \
  --annotations ~/ego4d_data/v2/annotations/av_train.json \
  --mel_cache outputs/mel_cache \
  --output_dir outputs/embeddings
```

### Loading embeddings in S6/S7

```python
import torch

# Load one clip's embedding
emb = torch.load("outputs/embeddings/{clip_uid}_{face_id}.pt")
# emb.shape == (T, 256)  — T windows, each a 256-d L2-normalized vector
# window_size = 80 frames = 0.8 s  |  stride = 16 frames = 0.1 s

# Load all clips via manifest
import csv
with open("outputs/embeddings/manifest.csv") as f:
    clips = list(csv.DictReader(f))
# cols: clip_uid, face_id, label, pt_path, duration_sec, n_windows
```

### Verify before consuming

```bash
python integration_test.py   # must print ALL INTEGRATION TESTS PASSED
```

### manifest.csv schema

| column | type | description |
|---|---|---|
| `clip_uid` | str | Ego4D clip identifier |
| `face_id` | int | Person track index within the clip |
| `label` | int | Ground-truth TTM (1 = talking to me, 0 = not) |
| `pt_path` | str | Absolute path to the `.pt` embedding file |
| `duration_sec` | float | Clip duration in seconds |
| `n_windows` | int | Number of temporal windows (= T dimension) |

## File structure

```
Audio_CNN/
├── configs/
│   ├── audio/           mel.yaml · model.yaml · train.yaml
│   └── experiment/      ablation_*.yaml
├── src/
│   ├── data/            mel_extractor · ttm_audio_dataset · augmentations · datamodule
│   ├── models/          audio_cnn · projection_head
│   ├── training/        lightning_module · callbacks
│   └── evaluation/      gradcam_visualizer · extract_embeddings
├── scripts/
│   ├── preprocess_audio.py
│   ├── train_audio_cnn.py
│   ├── evaluate.py
│   ├── visualize_gradcam.py
│   ├── dry_run.py
│   └── run_ablations.sh
└── environment.yaml
```
# ttm_audio_cnn
