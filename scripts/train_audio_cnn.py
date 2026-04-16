"""train_audio_cnn.py: Hydra entry point for training the TTM audio CNN.

Instantiates TTMAudioDataModule, AudioCNNModule, and Lightning Trainer from
the merged Hydra config (audio/train.yaml + experiment overrides).

After training completes, automatically runs extract_embeddings on the best
checkpoint unless skip_extraction=true or fast_dev_run is active.

Usage:
    # Standard run
    python scripts/train_audio_cnn.py

    # With experiment override
    python scripts/train_audio_cnn.py +experiment=ablation_no_mel

    # Fast dev run — 1 train batch + 1 val batch, no WandB, no extraction
    python scripts/train_audio_cnn.py trainer.fast_dev_run=true
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import TTMAudioDataModule
from src.training.callbacks import (
    EarlyStoppingOnPlateau,
    LRMonitor,
    WandBSpecgramLogger,
)
from src.training.lightning_module import AudioCNNModule

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="audio/train")
def main(cfg: DictConfig) -> None:
    """Full training + post-training embedding extraction.

    Args:
        cfg: Merged Hydra config dict (audio/mel + audio/model + audio/train).
    """
    pl.seed_everything(42, workers=True)

    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    annotation_path = Path(cfg.annotation_path)
    video_dir = Path(cfg.video_dir)
    mel_cache_dir = Path(cfg.mel_cache_dir)
    output_dir = Path(cfg.output_dir)

    is_fast_dev_run: bool = bool(cfg.get("trainer", {}).get("fast_dev_run", False))

    # ----------------------------------------------------------------
    # Shared mel / augment config dicts
    # ----------------------------------------------------------------
    mel_cfg = dict(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        clip_norm_range=cfg.clip_norm_range,
    )
    augment_cfg = dict(
        freq_mask_param=cfg.specaugment.freq_mask_param,
        time_mask_param=cfg.specaugment.time_mask_param,
        num_masks=cfg.specaugment.num_masks,
        noise_snr_range=tuple(cfg.noise_snr_range),
        pitch_shift_semitones=cfg.pitch_shift_semitones,
    )

    # ----------------------------------------------------------------
    # DataModule
    # ----------------------------------------------------------------
    datamodule = TTMAudioDataModule(
        annotation_path=annotation_path,
        video_dir=video_dir,
        mel_cache_dir=mel_cache_dir,
        mel_cfg=mel_cfg,
        augment_cfg=augment_cfg,
        batch_size=cfg.batch_size,
        num_workers=0 if is_fast_dev_run else cfg.num_workers,
        pin_memory=cfg.pin_memory and not is_fast_dev_run,
        max_frames=cfg.get("max_frames", None),
        write_cache=True,
    )
    # Setup now so we can pass class weights to the module
    datamodule.setup("fit")
    log.info(
        "Dataset sizes — train: %d  val: %d",
        len(datamodule.train_dataset),
        len(datamodule.val_dataset),
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    backbone_cfg = dict(
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
        # ablation_7x7_conv: swap (7,1)/(2,1) for (7,7)/(2,2)
        first_conv_kernel=tuple(cfg.get("first_conv_kernel", [7, 1])),
        first_conv_stride=tuple(cfg.get("first_conv_stride", [2, 1])),
    )
    head_cfg = dict(
        input_dim=256,
        hidden_dim=cfg.embedding_dim,
        output_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
    )

    module = AudioCNNModule(
        backbone_cfg=backbone_cfg,
        head_cfg=head_cfg,
        train_dataset=datamodule.train_dataset,   # only used for pos_weight
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_epochs=cfg.max_epochs,
        warmup_epochs=cfg.warmup_epochs,
        label_smoothing=cfg.label_smoothing,
        class_weight_mode=cfg.class_weight_mode,
        use_amp=cfg.use_amp,
    )

    # ----------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("experiment_name", "ttm_audio")
    run_name = f"{exp_name}_{timestamp}"

    ckpt_dir = output_dir / "checkpoints" / run_name
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/auc",
        mode="max",
        save_top_k=3,
        filename="epoch={epoch:02d}-val_auc={val/auc:.4f}",
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint_cb,
        LRMonitor(logging_interval="step"),
        EarlyStoppingOnPlateau(patience=10),
    ]
    # WandBSpecgramLogger requires a WandbLogger — skip for dry runs
    if not is_fast_dev_run:
        callbacks.append(WandBSpecgramLogger(num_samples=4, log_every_n_epochs=1))

    # ----------------------------------------------------------------
    # Logger — disable for fast_dev_run to avoid polluting WandB
    # ----------------------------------------------------------------
    if is_fast_dev_run:
        logger: Any = False
        log.info("fast_dev_run=True: WandB logging disabled.")
    else:
        wandb_logger = WandbLogger(
            project="ttm_audio",
            name=run_name,
            log_model=False,
            save_dir=str(output_dir / "wandb"),
        )
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        logger = wandb_logger

    # ----------------------------------------------------------------
    # Trainer
    # ----------------------------------------------------------------
    precision = "16-mixed" if cfg.use_amp else "32-true"
    extra_trainer_kwargs = {
        k: v for k, v in cfg.get("trainer", {}).items()
    }
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,
        enable_progress_bar=True,
        **extra_trainer_kwargs,
    )

    # ----------------------------------------------------------------
    # Train — DataModule provides both train and val loaders
    # ----------------------------------------------------------------
    trainer.fit(module, datamodule=datamodule)

    log.info("Training complete. Best checkpoint: %s", checkpoint_cb.best_model_path)

    # ----------------------------------------------------------------
    # Post-training: extract embeddings (skip for fast_dev_run)
    # ----------------------------------------------------------------
    if is_fast_dev_run:
        log.info("fast_dev_run: skipping embedding extraction.")
        return

    best_ckpt = Path(checkpoint_cb.best_model_path)
    if best_ckpt.exists() and not cfg.get("skip_extraction", False):
        log.info(
            "Run embedding extraction with:\n"
            "  python src/evaluation/extract_embeddings.py \\\n"
            "    --ckpt %s \\\n"
            "    --annotations %s \\\n"
            "    --mel_cache %s \\\n"
            "    --output_dir %s",
            best_ckpt,
            annotation_path,
            mel_cache_dir,
            output_dir / "embeddings",
        )


if __name__ == "__main__":
    main()
