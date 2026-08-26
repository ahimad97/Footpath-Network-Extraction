# Model Variants

`sidewalkformer/model.py` contains every integrated experiment branch. Select a
branch with `MODEL_TYPE` in the YAML config.

| Variant | `MODEL_TYPE` | Config | Segmentation/features | Graph construction | External weights |
|---|---|---|---|---|---|
| Paper SidewalkFormer | `segformer` | `config/sidewalkformer.yaml` | SegFormer B5 | SidewalkFormer GNN, symmetric edge scoring | Hugging Face backbone; trained `.ckpt` for inference |
| SAM-Road | `sam_road` | `config/baselines/sam_road_cityscale.yaml` | SAM ViT-B | SAM-Road TopoNet | SAM ViT-B checkpoint |
| SAM + SidewalkFormer topology | `sam_topo` | `config/baselines/sam_topo_cityscale.yaml` | SAM ViT-B | SidewalkFormer GNN | SAM ViT-B checkpoint |
| SegFormer + Tile2Net postprocess | `tile2net` | `config/baselines/tile2net_cityscale.yaml` | SegFormer | Tile2Net polygon/centerline postprocessing | Hugging Face backbone |
| Tile2Net + SidewalkFormer topology | `tile2net_topo` | `config/baselines/tile2net_topo_cityscale.yaml` | HRNet/OCR | SidewalkFormer GNN | Optional Tile2Net checkpoint |
| Tile2Net ablation (combined dataset) | `tile2net_topo` | `config/baselines/tile2net_topo_combined.yaml` | HRNet-W48/OCR | SidewalkFormer GNN | HRNet-W48 ImageNet weights |

## Config folders

| Folder | Contents |
|---|---|
| `config/` | `sidewalkformer.yaml`, the paper model |
| `config/baselines/` | SAM-Road, SAM-topology, Tile2Net, and Tile2Net-topology comparisons |

## Weight paths

Weight paths are portable relative placeholders beneath the ignored `weights/`
directory. They may also be changed to absolute paths on the execution
machine. No checkpoint or pretrained weight is part of the Git repository.

## Reproducibility rule

The paper config is the authoritative hyperparameter record. The training CLI
overrides (`--dataset-root`, `--output-dir`, `--run-name`, `--wandb-mode`)
affect only data location, artifact destination, run naming, and logging; they
do not alter the model, loss, thresholds, or graph schemas.
