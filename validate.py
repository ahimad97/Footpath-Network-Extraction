"""Evaluate a trained checkpoint on the held-out split of the configured dataset.

Reports segmentation IoU per class and edge (topology) metrics via
Lightning's ``Trainer.validate``.
"""

from argparse import ArgumentParser

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from sidewalkformer.dataset import SatMapDataset, custom_graph_collate_fn
from sidewalkformer.model import SidewalkFormer
from sidewalkformer.utils import cfg_get, load_config


PAPER_CONFIG = "config/sidewalkformer.yaml"


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PAPER_CONFIG)
    parser.add_argument("--checkpoint", required=True, help="Trained Lightning checkpoint (.ckpt).")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="External dataset root; overrides DATASET_ROOT in the YAML config.",
    )
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds the random node sampling of the validation graphs.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    if args.dataset_root:
        config.DATASET_ROOT = args.dataset_root

    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = bool(cfg_get(config, "CUDNN_BENCHMARK", False))
    torch.set_float32_matmul_precision("high")

    dataset = SatMapDataset(config, is_train=False, dev_run=False)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_graph_collate_fn,
        persistent_workers=int(config.DATA_WORKER_NUM) > 0,
    )

    precision = args.precision
    if torch.backends.mps.is_available() and precision == "bf16-mixed":
        precision = "32-true"

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        precision=precision,
    )
    trainer.validate(
        SidewalkFormer(config),
        dataloaders=loader,
        ckpt_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
