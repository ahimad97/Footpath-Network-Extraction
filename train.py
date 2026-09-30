"""Train SidewalkFormer or one of the baseline variants."""

from argparse import ArgumentParser
import os
from pathlib import Path
import random

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import torch
import wandb

from sidewalkformer.dataset import SatMapDataset, custom_graph_collate_fn
from sidewalkformer.model import SidewalkFormer
from sidewalkformer.utils import cfg_get, load_config


PAPER_CONFIG = "config/sidewalkformer.yaml"


def default_run_name(config_path: str, run_seed: int) -> str:
    """Build a stable, filesystem-safe name from the config and seed."""
    return f"{Path(config_path).stem}-seed{run_seed}"


def build_config_overrides(dataset_root: str | None) -> dict:
    """Return command-line overrides without changing the YAML file."""
    return {"DATASET_ROOT": dataset_root} if dataset_root else {}


def resolve_run_name(
    requested_name: str | None,
    config_path: str,
    run_seed: int,
    n_runs: int,
) -> str:
    if requested_name is None:
        return default_run_name(config_path, run_seed)
    if n_runs > 1:
        return f"{requested_name}-seed{run_seed}"
    return requested_name


def seed_worker(worker_id):
    """Seed numpy/random in each dataloader worker from the run seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _choice(value, allowed, default):
    return value if value in allowed else default


def train_once(config_path: str, run_seed: int, dev_run: bool, precision: str,
               resume_ckpt: str | None, config_overrides: dict | None = None,
               run_name: str | None = None, output_dir: str = "runs",
               wandb_project: str = "sidewalkformer",
               wandb_mode: str = "disabled"):
    """Train one model and return the path of its best (or last) checkpoint."""
    config = load_config(config_path)
    config.SEED = run_seed
    for k, v in (config_overrides or {}).items():
        config[k] = v

    pl.seed_everything(run_seed, workers=True)
    # cuDNN autotuning is off by default: it is non-deterministic and can fail
    # with "FIND was unable to find an engine" on some GPUs.
    torch.backends.cudnn.benchmark = bool(cfg_get(config, 'CUDNN_BENCHMARK', False))
    torch.set_float32_matmul_precision('high')

    generator = torch.Generator()
    generator.manual_seed(run_seed)
    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_graph_collate_fn,
        persistent_workers=(not dev_run and int(config.DATA_WORKER_NUM) > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_loader = torch.utils.data.DataLoader(
        SatMapDataset(config, is_train=True, dev_run=dev_run), shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(
        SatMapDataset(config, is_train=False, dev_run=dev_run), shuffle=False, **loader_kwargs)

    net = SidewalkFormer(config)

    if run_name is None:
        run_name = default_run_name(config_path, run_seed)
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    effective_wandb_mode = "disabled" if dev_run else wandb_mode
    wandb.init(name=run_name, project=wandb_project, config=config,
               dir=run_dir, mode=effective_wandb_mode)
    wandb_logger = WandbLogger(project=wandb_project, name=run_name, save_dir=run_dir,
                               config=config, mode=effective_wandb_mode)

    ckpt_monitor = cfg_get(config, 'CKPT_MONITOR', 'val_mask_loss')
    ckpt_mode = _choice(cfg_get(config, 'CKPT_MODE', 'min'), ('min', 'max'), 'min')
    early_stop_monitor = cfg_get(config, 'EARLY_STOP_MONITOR', ckpt_monitor)
    early_stop_mode = _choice(cfg_get(config, 'EARLY_STOP_MODE', ckpt_mode), ('min', 'max'), ckpt_mode)
    print(f"[train] ckpt_monitor={ckpt_monitor} ckpt_mode={ckpt_mode} "
          f"early_stop_monitor={early_stop_monitor} early_stop_mode={early_stop_mode}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
        filename="{epoch:02d}-valloss{val_loss:.3f}",
    )
    early_stop_cb = EarlyStopping(
        monitor=early_stop_monitor,
        mode=early_stop_mode,
        patience=int(cfg_get(config, 'EARLY_STOP_PATIENCE', 7)),
        verbose=True,
    )

    effective_precision = precision
    if torch.backends.mps.is_available() and precision == "bf16-mixed":
        print("[train] MPS detected: using precision '32-true' instead of 'bf16-mixed'.")
        effective_precision = "32-true"

    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        devices="auto",
        gradient_clip_val=0.5,
        max_epochs=int(cfg_get(config, 'TRAIN_EPOCHS', 25)),
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval='step'), early_stop_cb],
        logger=wandb_logger,
        fast_dev_run=dev_run,
        accumulate_grad_batches=1,
        precision=effective_precision,
        default_root_dir=run_dir,
    )
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume_ckpt)

    best_ckpt = checkpoint_cb.best_model_path
    last_ckpt = checkpoint_cb.last_model_path
    print(f"[train] Finished. best_ckpt={best_ckpt} last_ckpt={last_ckpt}")
    wandb.finish()
    torch.cuda.empty_cache()
    return best_ckpt or last_ckpt or None


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PAPER_CONFIG)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="External dataset root; overrides DATASET_ROOT in the YAML config.",
    )
    parser.add_argument("--output-dir", default="runs",
                        help="Checkpoints and logs are written to <output-dir>/<run-name>/.")
    parser.add_argument("--run-name", default=None,
                        help="Defaults to <config name>-seed<seed>.")
    parser.add_argument("--wandb-project", default="sidewalkformer")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="Weights & Biases mode. Disabled by default for credential-free runs.",
    )
    parser.add_argument("--resume", default=None, help="Lightning checkpoint to resume from.")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--fast_dev_run", default=False, action="store_true",
                        help="Run one train/val batch on four tiles as a smoke test.")
    parser.add_argument("--dev_run", default=False, action="store_true",
                        help="Alias of --fast_dev_run.")
    parser.add_argument("--n_runs", type=int, default=1, help="Train this many seeds in sequence.")
    parser.add_argument("--base_seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dev_run = args.dev_run or args.fast_dev_run
    config_overrides = build_config_overrides(args.dataset_root)
    for k in range(args.n_runs):
        run_seed = args.base_seed + k
        train_once(
            args.config,
            run_seed,
            dev_run,
            args.precision,
            args.resume,
            config_overrides=config_overrides,
            run_name=resolve_run_name(args.run_name, args.config, run_seed, args.n_runs),
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            wandb_mode=args.wandb_mode,
        )


if __name__ == "__main__":
    main()
