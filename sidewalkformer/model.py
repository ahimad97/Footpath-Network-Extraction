"""SidewalkFormer and the integrated baseline variants.

``MODEL_TYPE`` in the config selects the variant:

* ``segformer``      SegFormer segmentation + SidewalkFormer topology GNN (paper model)
* ``tile2net``       SegFormer segmentation only; the graph comes from Tile2Net
                     post-processing at inference time
* ``tile2net_topo``  Tile2Net HRNet-OCR backbone + SidewalkFormer topology GNN
* ``sam_road``       SAM-Road (SAM ViT encoder + SAM-Road TopoNet)
* ``sam_topo``       SAM ViT encoder + SidewalkFormer topology GNN
"""

import math
import os
import random
from contextlib import nullcontext
from functools import partial

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GraphNorm,
    LayerNorm,
    SAGEConv,
    TransformerConv,
)
from torchmetrics import ConfusionMatrix, Precision, Recall
from torchmetrics.classification import (
    AveragePrecision,
    F1Score,
    JaccardIndex,
    PrecisionRecallCurve,
)
from transformers import SegformerDecodeHead, SegformerForSemanticSegmentation

from .utils import cfg_get

try:
    from sam_road.model import SAMRoad
except ImportError as exc:
    SAMRoad = None
    print(f"[warning] SAM-Road baseline unavailable: {exc}")

try:
    from tile2net.tileseg.config import cfg as t2n_cfg
    from tile2net.tileseg.network.ocrnet import MscaleOCR, OCRNet
    _TILE2NET_OK = True
except Exception as exc:
    _TILE2NET_OK = False
    print(f"[warning] Tile2Net baseline unavailable: {exc}")


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# IMAGE_EMB_MODE -> (number of last SegFormer stages fused, stage whose
# resolution the fused map takes). ``fuse2_4`` fuses the 1/16 and 1/32 stages
# at 1/4 resolution. Any other value uses the last stage unchanged.
FUSE_MODES = {
    "fuse4_16": (4, -2),
    "fuse4_4": (4, -4),
    "fuse2_16": (2, -2),
    "fuse2_4": (2, -4),
}


def _imagenet_to_rgb255(pixel_values):
    """Undo ImageNet normalisation: [B,3,H,W] normalised -> RGB in 0..255."""
    mean = torch.tensor(_IMAGENET_MEAN, device=pixel_values.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=pixel_values.device).view(1, 3, 1, 1)
    return (pixel_values * std + mean) * 255.0


def _to_grid_coords(pts, Hf, Wf, align_corners=False):
    """Feature-map pixel coords [..., 2] -> ``grid_sample`` coords in [-1, 1]."""
    x, y = pts[..., 0], pts[..., 1]
    if align_corners:
        gx = x / (Wf - 1) * 2 - 1
        gy = y / (Hf - 1) * 2 - 1
    else:
        gx = (x + 0.5) / Wf * 2 - 1
        gy = (y + 0.5) / Hf * 2 - 1
    return torch.stack([gx, gy], dim=-1)


class PairSetHead(nn.Module):
    """Contextual edge scorer.

    Each candidate edge ``u -> v`` becomes a pair token ``[z_u, z_v, edge_attr]``.
    Tokens that share a source node attend to each other, so a node's
    candidate links are scored jointly rather than independently.

    Inputs: ``z`` [N, D] node embeddings, ``edge_index`` [2, E] candidate
    edges, ``edge_attr`` [E, A] or None. Output: logits [E].
    """

    def __init__(self, d_node, edge_attr_dim=2, d_pair=128, n_layers=1,
                 n_heads=1, dropout=0.1, use_edge_attr=True):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        in_dim = 2 * d_node + (edge_attr_dim if use_edge_attr else 0)
        self.proj = nn.Linear(in_dim, d_pair)
        layer = nn.TransformerEncoderLayer(
            d_model=d_pair, nhead=n_heads, dim_feedforward=4 * d_pair,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_pair, 1)

    def forward(self, z, edge_index, edge_attr=None):
        u, v = edge_index
        if u.numel() == 0:
            return z.new_zeros((0,))
        feats = [z[u], z[v]]
        if self.use_edge_attr and edge_attr is not None:
            feats.append(edge_attr)
        h = F.relu(self.proj(torch.cat(feats, dim=-1)))               # [E, d_pair]

        # Group edges by source node, pad to [num_sources, max_degree, d_pair].
        sort_idx = torch.argsort(u)
        _, counts = torch.unique_consecutive(u[sort_idx], return_counts=True)
        counts = counts.tolist()
        H_pad = pad_sequence(list(torch.split(h[sort_idx], counts)), batch_first=True)
        padding = torch.ones(H_pad.shape[:2], dtype=torch.bool, device=H_pad.device)
        for b, length in enumerate(counts):
            padding[b, :length] = False

        logits_pad = self.out(self.encoder(H_pad, src_key_padding_mask=padding)).squeeze(-1)

        # Unpad back to the original edge order.
        logits_sorted = H_pad.new_empty((h.size(0),))
        offset = 0
        for b, length in enumerate(counts):
            logits_sorted[offset:offset + length] = logits_pad[b, :length]
            offset += length
        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        return logits_sorted[inv_sort]


class PatchSampler(nn.Module):
    """Sample image features at node locations.

    With ``grid_size == 1`` this is a single bilinear sample per node. With
    ``grid_size > 1`` a KxK grid of offsets (in feature-map pixels) around each
    node is sampled and pooled with learned attention.
    """

    def __init__(self, feat_dim: int, grid_size: int = 1, align_corners: bool = False):
        super().__init__()
        self.grid_size = grid_size
        self.align_corners = align_corners
        if grid_size > 1:
            half = grid_size // 2
            offsets = torch.stack(torch.meshgrid(
                torch.arange(-half, half + 1, dtype=torch.float32),
                torch.arange(-half, half + 1, dtype=torch.float32),
                indexing='xy'), dim=-1).reshape(-1, 2)                    # [K*K, 2] (dx, dy)
            self.register_buffer('offsets', offsets)
            self.attn_proj = nn.Linear(feat_dim, 1)

    def forward(self, feature_maps, pts_fm):
        """feature_maps [B,C,Hf,Wf], pts_fm [B,N,2] feature-map pixels -> [B,N,C]."""
        B, C, Hf, Wf = feature_maps.shape
        if self.grid_size <= 1:
            grid = _to_grid_coords(pts_fm, Hf, Wf, self.align_corners).unsqueeze(2)
            samp = F.grid_sample(feature_maps, grid, mode='bilinear',
                                 align_corners=self.align_corners)
            return samp.squeeze(-1).permute(0, 2, 1)

        K2 = self.offsets.shape[0]
        pts = (pts_fm.unsqueeze(2) + self.offsets[None, None]).reshape(B, -1, 2)  # [B, N*K2, 2]
        grid = _to_grid_coords(pts, Hf, Wf, self.align_corners).unsqueeze(2)
        samp = F.grid_sample(feature_maps, grid, mode='bilinear',
                             align_corners=self.align_corners)            # [B, C, N*K2, 1]
        samp = samp.squeeze(-1).permute(0, 2, 1).reshape(B, -1, K2, C)    # [B, N, K2, C]
        attn = torch.softmax(self.attn_proj(samp).squeeze(-1), dim=-1)    # [B, N, K2]
        return (attn.unsqueeze(-1) * samp).sum(dim=2)


class SinusoidalPosEnc2D(nn.Module):
    """Sinusoidal encoding of (x, y) patch coordinates followed by a linear layer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        n_bands = d_model // 4
        freqs = torch.exp(torch.arange(n_bands, dtype=torch.float32)
                          * -(math.log(10000.0) / max(n_bands - 1, 1)))
        self.register_buffer('freqs', freqs)
        self.proj = nn.Linear(n_bands * 4, d_model)

    def forward(self, pos, patch_size):
        """pos [N, 2] in patch pixels -> [N, d_model]."""
        p = pos / patch_size
        x, y = p[:, 0:1], p[:, 1:2]
        freqs = self.freqs.unsqueeze(0)
        enc = torch.cat([
            torch.sin(x * freqs), torch.cos(x * freqs),
            torch.sin(y * freqs), torch.cos(y * freqs),
        ], dim=-1)
        return self.proj(enc)


class EdgeVisualEncoder(nn.Module):
    """Pool image features sampled at K points along each edge."""

    def __init__(self, feat_dim: int, edge_out_dim: int = 32, n_samples: int = 5):
        super().__init__()
        self.n_samples = n_samples
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, edge_out_dim),
            nn.ReLU(),
            nn.Linear(edge_out_dim, edge_out_dim),
        )

    def forward(self, feature_maps, pos, edge_index, patch_size, Hf, Wf):
        """feature_maps [1,C,Hf,Wf], pos [N,2] patch pixels, edge_index [2,E] -> [E, out]."""
        E = edge_index.shape[1]
        if E == 0:
            return feature_maps.new_zeros((0, self.mlp[-1].out_features))

        src, dst = edge_index
        p0, p1 = pos[src], pos[dst]
        # Interior points only; the endpoints are covered by the node features.
        K = self.n_samples
        t = torch.linspace(0.0, 1.0, K + 2, device=pos.device)[1:-1].view(1, K, 1)
        mid_pts = p0.unsqueeze(1) * (1 - t) + p1.unsqueeze(1) * t          # [E, K, 2]

        W_in = max(float(patch_size), pos[:, 0].max().item() + 1.0)
        H_in = max(float(patch_size), pos[:, 1].max().item() + 1.0)
        mid_fm = mid_pts.clone()
        mid_fm[..., 0] *= (Wf - 1) / (W_in - 1)
        mid_fm[..., 1] *= (Hf - 1) / (H_in - 1)

        grid = _to_grid_coords(mid_fm, Hf, Wf).reshape(1, E * K, 1, 2)
        samp = F.grid_sample(feature_maps, grid, mode='bilinear', align_corners=False)
        samp = samp.squeeze(-1).squeeze(0).T.reshape(E, K, -1)               # [E, K, C]
        return self.mlp(samp.mean(dim=1))


class SegRefinementHead(nn.Module):
    """Residual conv head that refines segmentation logits with a topology canvas."""

    def __init__(self, num_classes: int, hidden_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_classes + 1, hidden_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, num_classes, 1)

    def forward(self, seg_logits, topo_canvas):
        topo_canvas = topo_canvas.to(dtype=seg_logits.dtype)
        x = torch.cat([seg_logits, topo_canvas], dim=1)
        return seg_logits + self.conv2(F.relu(self.conv1(x)))


def _rasterize_topo_canvas(pos, edge_index, edge_scores, batch_ids,
                           B, H, W, n_pts=10, sigma=3.0):
    """Differentiably splat predicted edges onto a [B, 1, H, W] canvas.

    Each edge deposits its score at ``n_pts`` points along the segment; the
    canvas is then Gaussian-blurred. Gradients flow through ``edge_scores``.
    """
    device = pos.device
    # Accumulate in fp32; index_add_ needs matching dtypes under AMP.
    canvas = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    if edge_index.numel() == 0 or edge_scores.numel() == 0:
        return canvas

    src, dst = edge_index
    p0, p1 = pos[src], pos[dst]
    t = torch.linspace(0.0, 1.0, n_pts, device=device).view(1, -1, 1)
    pts_flat = (p0.unsqueeze(1) * (1 - t) + p1.unsqueeze(1) * t).reshape(-1, 2)
    scores_flat = edge_scores.unsqueeze(1).expand(-1, n_pts).reshape(-1).to(
        device=device, dtype=canvas.dtype)
    ids_flat = batch_ids[src].unsqueeze(1).expand(-1, n_pts).reshape(-1)

    xi = pts_flat[:, 0].long().clamp(0, W - 1)
    yi = pts_flat[:, 1].long().clamp(0, H - 1)
    linear_idx = ids_flat * H * W + yi * W + xi
    canvas_flat = canvas.view(-1)
    canvas_flat.index_add_(0, linear_idx.long(), scores_flat)
    canvas = canvas_flat.view(B, 1, H, W)

    if sigma > 0:
        k = int(2 * sigma + 1) | 1
        pad = k // 2
        kernel = torch.arange(k, dtype=torch.float32, device=device) - pad
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()
        canvas = F.conv2d(F.pad(canvas, [pad, pad, 0, 0], mode='reflect'), kernel.view(1, 1, 1, -1))
        canvas = F.conv2d(F.pad(canvas, [0, 0, pad, pad], mode='reflect'), kernel.view(1, 1, -1, 1))
    return canvas.to(dtype=edge_scores.dtype)


class TopoNetGNN(nn.Module):
    """Graph network that scores candidate edges between node proposals.

    Expected PyG ``Data`` fields:
      x                 [N, D]       node features sampled from the image
      pos               [N, 2]       node coordinates in patch pixels
      edge_index        [2, E_msg]   directed message-passing edges (both directions)
      edge_attr         [E_msg, 2]   (dx, dy) per message-passing edge
      edge_label_index  [2, E_pred]  one column per unordered candidate edge
      edge_label_attr   [E_pred, 2]  (dx, dy) per candidate edge

    With ``undirected_edge_pred`` each candidate is scored in both directions
    and the two logits are averaged, so the prediction is symmetric.
    """

    def __init__(
        self,
        in_node_dim,                     # 0 -> use node coordinates as input features
        hidden_dim=128,
        num_layers=3,
        heads=4,
        conv_type="transformer",         # 'transformer' | 'gat' | 'gatv2' | 'gcn' | 'sage'
        encoder_type="gnn",              # 'gnn' | 'mlp' (no message passing)
        use_edge_attr=True,
        edge_enc_dim=32,
        dropout=0.1,
        decoder_type="pair",             # 'pair' (PairSetHead) | anything else -> MLP
        pair_d=128,
        pair_layers=2,
        pair_heads=2,
        neighbor_radius=None,
        use_pos_enc: bool = False,
        patch_size: int = 512,
        edge_visual_dim: int = 0,
        undirected_edge_pred: bool = True,
    ):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.decoder_type = decoder_type.lower()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.conv_type = conv_type.lower()
        self.encoder_type = encoder_type.lower()
        self.edge_visual_dim = edge_visual_dim
        self.undirected_edge_pred = undirected_edge_pred

        self.pos_enc = SinusoidalPosEnc2D(hidden_dim) if use_pos_enc else None
        self.patch_size = float(patch_size)

        self.use_pos_as_feat = in_node_dim == 0
        self.proj = nn.Linear(2 if self.use_pos_as_feat else in_node_dim, hidden_dim)

        # Edge encoder: (dx, dy [, visual features]) -> edge_enc_dim
        if use_edge_attr:
            self.edge_enc = nn.Sequential(
                nn.Linear(2 + edge_visual_dim, edge_enc_dim),
                nn.ReLU(),
                nn.Linear(edge_enc_dim, edge_enc_dim),
            )
            edge_dim_for_conv = edge_enc_dim
        else:
            self.edge_enc = None
            edge_dim_for_conv = None

        convs, norms = [], []
        for _ in range(num_layers):
            if self.conv_type == "transformer":
                conv = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                       dropout=dropout, edge_dim=edge_dim_for_conv)
                norm = LayerNorm(hidden_dim)
            elif self.conv_type == "gat":
                conv = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout,
                               edge_dim=edge_dim_for_conv, add_self_loops=False)
                norm = LayerNorm(hidden_dim)
            elif self.conv_type == "gatv2":
                conv = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout,
                                 edge_dim=edge_dim_for_conv, add_self_loops=False)
                norm = LayerNorm(hidden_dim)
            elif self.conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
                norm = GraphNorm(hidden_dim)
            elif self.conv_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim, project=True)
                norm = GraphNorm(hidden_dim)
            else:
                raise ValueError(f"Unknown conv_type {conv_type}")
            convs.append(conv)
            norms.append(norm)
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        if self.encoder_type == "mlp":
            mlp = []
            for _ in range(num_layers):
                mlp += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            self.node_mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        else:
            self.node_mlp = None

        # Independent MLP edge decoder (used when decoder_type != 'pair').
        dec_in = hidden_dim * 2 + (edge_enc_dim if use_edge_attr else 0)
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_head = PairSetHead(
            d_node=hidden_dim,
            edge_attr_dim=2 + edge_visual_dim,
            d_pair=pair_d,
            n_layers=pair_layers,
            n_heads=pair_heads,
            dropout=0.1,
            use_edge_attr=use_edge_attr,
        )

    def _compose_raw_edge(self, data):
        """Message-passing edge features: (dx, dy) plus optional visual features."""
        ea = data.edge_attr
        if self.edge_visual_dim > 0 and getattr(data, "edge_visual_feat", None) is not None:
            ea = torch.cat([ea, data.edge_visual_feat], dim=-1)
        return ea

    def encode_nodes(self, data):
        """Return node embeddings [N, hidden_dim]."""
        if int(data.num_nodes) == 0:
            return data.pos.new_zeros((0, self.hidden_dim))

        x = F.relu(self.proj(data.pos if self.use_pos_as_feat else data.x))
        if self.pos_enc is not None:
            x = x + self.pos_enc(data.pos, self.patch_size)

        enc_edge = None
        if self.use_edge_attr and getattr(data, "edge_attr", None) is not None:
            enc_edge = self.edge_enc(self._compose_raw_edge(data))

        if self.encoder_type == "mlp":
            return self.node_mlp(x)

        no_edges = data.edge_index is None or data.edge_index.numel() == 0
        for conv, norm in zip(self.convs, self.norms):
            if no_edges:
                h = x
            elif isinstance(conv, (TransformerConv, GATConv)):
                h = conv(x, data.edge_index, enc_edge)
            else:
                h = conv(x, data.edge_index)
            x = F.relu(norm(x + self.dropout(h)))
        return x

    def decode_edges(self, z, edge_index, edge_attr=None):
        """MLP decoder: z [N, H], edge_index [2, E], raw edge_attr [E, A] -> logits [E]."""
        if edge_index.numel() == 0:
            return z.new_zeros((0,))
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=-1)
        if self.use_edge_attr and edge_attr is not None:
            h = torch.cat([h, self.edge_enc(edge_attr)], dim=-1)
        return self.decoder(h).squeeze(-1)

    @staticmethod
    def _reverse_edge_attr(edge_attr):
        if edge_attr is None:
            return None
        rev_attr = edge_attr.clone()
        # Negate the spatial offset; visual features are shared by both directions.
        rev_attr[:, :2] = -rev_attr[:, :2]
        return rev_attr

    def _label_edge_attr(self, data):
        """Raw features for the candidate edges in ``edge_label_index``."""
        edge_attr = getattr(data, "edge_label_attr", None)
        if edge_attr is None:
            edge_attr = getattr(data, "edge_attr", None)
            if edge_attr is not None and edge_attr.size(0) != data.edge_label_index.size(1):
                src, dst = data.edge_label_index
                edge_attr = data.pos[dst] - data.pos[src]
        if edge_attr is not None and self.edge_visual_dim > 0:
            visual = getattr(data, "edge_label_visual_feat", None)
            if visual is None:
                visual = getattr(data, "edge_visual_feat", None)
                if visual is not None and visual.size(0) != data.edge_label_index.size(1):
                    visual = None
            if visual is not None:
                edge_attr = torch.cat([edge_attr, visual], dim=-1)
        return edge_attr

    def decode_edges_pairset(self, z, data):
        if data.edge_label_index.numel() == 0:
            return z.new_zeros((0,))
        edge_index = data.edge_label_index
        ea = self._label_edge_attr(data) if self.use_edge_attr else None
        if not self.undirected_edge_pred:
            return self.pair_head(z=z, edge_index=edge_index, edge_attr=ea)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        if ea is not None:
            ea = torch.cat([ea, self._reverse_edge_attr(ea)], dim=0)
        logits = self.pair_head(z=z, edge_index=edge_index, edge_attr=ea)
        E = data.edge_label_index.size(1)
        return 0.5 * (logits[:E] + logits[E:])

    def forward(self, data):
        """Return (logits [E], probabilities [E]) for ``data.edge_label_index``."""
        z = self.encode_nodes(data)
        if self.decoder_type == 'pair':
            logits = self.decode_edges_pairset(z, data)
        else:
            edge_index = data.edge_label_index
            edge_attr = self._label_edge_attr(data) if self.use_edge_attr else None
            if self.undirected_edge_pred and edge_index.numel() > 0:
                logits_fwd = self.decode_edges(z, edge_index, edge_attr)
                logits_rev = self.decode_edges(z, edge_index.flip(0), self._reverse_edge_attr(edge_attr))
                logits = 0.5 * (logits_fwd + logits_rev)
            else:
                logits = self.decode_edges(z, edge_index, edge_attr)
        return logits, torch.sigmoid(logits)


class ScaleFuser(nn.Module):
    """Fuse encoder stages at a target resolution with learned per-channel gates.

    Each stage is projected to ``out_ch`` (1x1 conv + GroupNorm + GELU) and
    resized to ``target_hw``; a softmax over stages mixes them per channel.
    """

    def __init__(self, in_chs, out_ch=256, use_gn=True):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_ch, 1, bias=False),
                nn.GroupNorm(32, out_ch) if use_gn else nn.Identity(),
                nn.GELU(),
            ) for c in in_chs
        ])
        self.alpha = nn.Parameter(torch.zeros(len(in_chs), out_ch))

    def forward(self, feats, target_hw):
        resized = [
            F.interpolate(p(f), size=target_hw, mode='bilinear', align_corners=False, antialias=True)
            for f, p in zip(feats, self.proj)
        ]
        weights = torch.softmax(self.alpha, dim=0)[:, None, :, None, None]   # [S, 1, C, 1, 1]
        return (weights * torch.stack(resized, dim=0)).sum(dim=0)


def _no_weight_decay(name, param):
    return param.ndim == 1 or any(k in name.lower() for k in ("bias", "norm", "bn", "layernorm"))


class SidewalkFormer(pl.LightningModule):
    """Joint segmentation + topology model (see module docstring for variants)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_type = config.MODEL_TYPE

        # The SF/combined datasets merge zebra crossings into ``crossing``
        # (4 classes); CityScale keeps them separate (5 classes).
        dataset_loc = str(cfg_get(config, 'dataset_location',
                                  cfg_get(config, 'DATASET', 'cityscale'))).strip().lower()
        if dataset_loc in {"sf", "sanfrancisco", "san_francisco", "san-francisco", "combined"}:
            self.id2label = {0: 'background', 1: 'sidewalk', 2: 'road', 3: 'crossing'}
        else:
            self.id2label = {0: 'background', 1: 'sidewalk', 2: 'road', 3: 'crossing',
                             4: 'zebra crossing'}
        self.num_classes = len(self.id2label)
        self.config.NUM_CLASSES = self.num_classes  # read by SAM-Road
        self.label2id = {v: k for k, v in self.id2label.items()}

        on_mps = torch.backends.mps.is_available()
        if model_type in ('segformer', 'tile2net'):
            self.initialize_segformer()
            # BatchNorm backward is unreliable on Apple MPS.
            if on_mps and cfg_get(config, 'MPS_REPLACE_BATCHNORM', True):
                self._replace_batchnorm_with_groupnorm(self.decoder)
            if on_mps and cfg_get(config, 'MPS_FREEZE_BATCHNORM', True):
                self._freeze_batchnorm_modules(self.decoder)
        elif model_type in ('sam_road', 'sam_topo'):
            if SAMRoad is None:
                raise ImportError("SAM-Road could not be imported from third_party/sam_road.")
            self.sam_road_model = SAMRoad(config)
            self.encoder_output_dim = 256  # SAM image-embedding channels
        elif model_type == 'tile2net_topo':
            if not _TILE2NET_OK:
                raise ImportError("Tile2Net (HRNet-OCR) could not be imported from third_party/tile2net.")
            self.initialize_tile2net()
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")

        self._seg_only = cfg_get(config, 'SEG_ONLY', False)
        self._balance_loss = cfg_get(config, 'balance_loss', False)
        self._mask_loss_weight = float(cfg_get(config, 'MASK_LOSS_WEIGHT', 1.0))
        self._topo_loss_weight = float(cfg_get(config, 'TOPO_LOSS_WEIGHT', 1.0))
        self._topo_warmup_fraction = float(cfg_get(config, 'TOPO_WARMUP_FRACTION', 0.0))
        self._freeze_segmentation = bool(cfg_get(config, 'FREEZE_SEGMENTATION', False))
        # Stop topology gradients from reaching the SegFormer encoder (the fuser
        # still trains on them).
        self._detach_topo_features = bool(cfg_get(config, 'DETACH_TOPO_FEATURES', True))

        if model_type == 'segformer':
            self.fuse_mode = cfg_get(config, "IMAGE_EMB_MODE", "last")
            if self.fuse_mode in FUSE_MODES:
                n_stages, _ = FUSE_MODES[self.fuse_mode]
                out_ch = cfg_get(config, "EMB_OUT_CH", 512)
                self.fuser = ScaleFuser(list(self.encoder.config.hidden_sizes)[-n_stages:], out_ch)
                self.encoder_output_dim = out_ch
                if self._freeze_segmentation:
                    for param in self.fuser.parameters():
                        param.requires_grad = False
            else:
                self.fuser = None
                self.encoder_output_dim = self.encoder.config.hidden_sizes[-1]

        self._node_grid = int(cfg_get(config, 'NODE_SAMPLE_GRID', 1))
        self.node_sampler = PatchSampler(
            feat_dim=self.encoder_output_dim if self._node_grid > 1 else 0,
            grid_size=self._node_grid,
        )

        self._use_edge_visual = bool(cfg_get(config, 'EDGE_VISUAL_FEAT', False))
        edge_visual_dim = 0
        if self._use_edge_visual:
            edge_visual_dim = 32
            self.edge_visual_enc = EdgeVisualEncoder(
                feat_dim=self.encoder_output_dim, edge_out_dim=edge_visual_dim, n_samples=5)

        if model_type not in ('sam_road', 'tile2net') and not self._seg_only:
            self.topo_net = TopoNetGNN(
                in_node_dim=self.encoder_output_dim,
                hidden_dim=cfg_get(config, 'hidden_dim', 128),
                heads=cfg_get(config, 'heads', 4),
                num_layers=cfg_get(config, 'num_layers', 3),
                conv_type=cfg_get(config, 'conv_type', 'transformer'),
                encoder_type=cfg_get(config, 'encoder_type', 'gnn'),
                use_edge_attr=cfg_get(config, 'use_edge_attr', True),
                decoder_type=cfg_get(config, 'decoder_type', 'pair'),
                use_pos_enc=bool(cfg_get(config, 'NODE_POS_ENC', False)),
                patch_size=int(config.PATCH_SIZE),
                edge_visual_dim=edge_visual_dim,
                undirected_edge_pred=bool(cfg_get(config, 'UNDIRECTED_EDGE_PRED', True)),
            )

        self._use_seg_refine = bool(cfg_get(config, 'SEG_REFINEMENT', False))
        if self._use_seg_refine:
            self.seg_refine = SegRefinementHead(self.num_classes)

        # Auxiliary topology losses (0 disables them):
        #   CONN_LOSS_WEIGHT      path-sampling soft-connectivity loss
        #   LAPLACIAN_LOSS_WEIGHT hinge on the Fiedler value of the soft Laplacian
        self._conn_loss_weight = float(cfg_get(config, 'CONN_LOSS_WEIGHT', 0.0))
        self._laplacian_loss_weight = float(cfg_get(config, 'LAPLACIAN_LOSS_WEIGHT', 0.0))
        # Extra weight on reference edges that are bridges (their removal
        # disconnects the local graph). 1.0 disables the reweighting.
        self._bridge_edge_weight = float(cfg_get(config, 'BRIDGE_EDGE_WEIGHT', 3.0))

        if cfg_get(config, 'FOCAL_LOSS_SEGMENTATION', False):
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            ce_weights = {
                4: [1.0, 4.0, 2.0, 4.0],
                5: [1.0, 4.0, 2.0, 4.0, 4.0],
            }.get(self.num_classes, [1.0] * self.num_classes)
            self.register_buffer("ce_weights", torch.tensor(ce_weights, dtype=torch.float))
            self.mask_criterion = nn.CrossEntropyLoss(weight=self.ce_weights, ignore_index=-100)
        # Total segmentation loss = CE + DICE_LOSS_WEIGHT * Dice (background excluded).
        self._use_dice_loss = bool(cfg_get(config, 'USE_DICE_LOSS', False))
        self._dice_loss_weight = float(cfg_get(config, 'DICE_LOSS_WEIGHT', 0.5))
        self.topo_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(config.EDGE_POS_WEIGHT)]), reduction='none')

        # Validation metrics.
        self.iou_metric = JaccardIndex(num_classes=self.num_classes, task='multiclass',
                                       average='none', ignore_index=0)
        self.topo_f1 = F1Score(task='binary', threshold=0.5)
        self.edge_prec = Precision(task="binary", threshold=0.5)
        self.edge_rec = Recall(task="binary", threshold=0.5)
        self.edge_cm = ConfusionMatrix(num_classes=2, task="binary")
        self.edge_ap = AveragePrecision(task='binary')
        self.edge_prc = PrecisionRecallCurve(task='binary', thresholds=100)
        self.register_buffer("val_pixel_hist", torch.zeros(self.num_classes, dtype=torch.long))
        # Learned task-uncertainty weights [seg, topo], used when balance_loss is set.
        self.log_vars = nn.Parameter(torch.zeros(2))

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _replace_batchnorm_with_groupnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                channels = child.num_features
                new_norm = nn.GroupNorm(32 if channels % 32 == 0 else 1, channels, affine=True)
                if getattr(child, "affine", False):
                    with torch.no_grad():
                        new_norm.weight.copy_(child.weight)
                        new_norm.bias.copy_(child.bias)
                setattr(module, name, new_norm)
            else:
                SidewalkFormer._replace_batchnorm_with_groupnorm(child)

    @staticmethod
    def _freeze_batchnorm_modules(module):
        for m in module.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def initialize_segformer(self):
        """Load the Hugging Face SegFormer and replace its head for our classes."""
        full_model = SegformerForSemanticSegmentation.from_pretrained(
            self.config.model_name, output_hidden_states=True)
        full_model.config.num_labels = self.num_classes
        full_model.config.id2label = self.id2label
        full_model.config.label2id = self.label2id
        full_model.decode_head = SegformerDecodeHead(full_model.config)

        self.encoder = full_model.segformer
        self.decoder = full_model.decode_head

        if cfg_get(self.config, 'FREEZE_SEGMENTATION', False):
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif cfg_get(self.config, 'FREEZE_ENCODER', False):
            for param in self.encoder.parameters():
                param.requires_grad = False

        seg_ckpt = cfg_get(self.config, 'SEGFORMER_PRETRAINED_CKPT', None)
        if seg_ckpt:
            self._load_segformer_pretrained(seg_ckpt)

    def _load_segformer_pretrained(self, ckpt_path: str):
        """Warm-start encoder/decoder from a segmentation-only Lightning checkpoint.

        Only ``encoder.*`` and ``decoder.*`` keys are transferred; keys with a
        shape mismatch are skipped.
        """
        if not os.path.isfile(ckpt_path):
            print(f"[warning] SEGFORMER_PRETRAINED_CKPT not found: {ckpt_path}; skipping.")
            return

        print(f"[model] Loading pretrained SegFormer weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in state_dict.items():
            if not k.startswith(('encoder.', 'decoder.')):
                continue
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
            else:
                skipped += 1
                if k in own_state:
                    print(f"  [skip] {k}: shape mismatch "
                          f"(ckpt={list(v.shape)}, model={list(own_state[k].shape)})")
                else:
                    print(f"  [skip] {k}: not in current model")
        self.load_state_dict(own_state, strict=False)
        print(f"[model] Loaded {loaded} SegFormer tensors, skipped {skipped}")

    def initialize_tile2net(self):
        """Build Tile2Net's HRNet-W48 + OCR (+ hierarchical multi-scale attention)."""
        t2n_cfg.immutable(False)
        t2n_cfg.DATASET.NUM_CLASSES = self.num_classes
        t2n_cfg.MODEL.OCR.MID_CHANNELS = 512
        t2n_cfg.MODEL.OCR.KEY_CHANNELS = 256

        # HRNet-W48 stage configuration (Tile2Net defaults).
        t2n_cfg.MODEL.OCR_EXTRA.STAGE1.NUM_MODULES = 1
        t2n_cfg.MODEL.OCR_EXTRA.STAGE1.NUM_BLOCKS = [4]
        t2n_cfg.MODEL.OCR_EXTRA.STAGE1.NUM_CHANNELS = [64]
        t2n_cfg.MODEL.OCR_EXTRA.STAGE2.NUM_MODULES = 1
        t2n_cfg.MODEL.OCR_EXTRA.STAGE2.NUM_BRANCHES = 2
        t2n_cfg.MODEL.OCR_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
        t2n_cfg.MODEL.OCR_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
        t2n_cfg.MODEL.OCR_EXTRA.STAGE3.NUM_MODULES = 4
        t2n_cfg.MODEL.OCR_EXTRA.STAGE3.NUM_BRANCHES = 3
        t2n_cfg.MODEL.OCR_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
        t2n_cfg.MODEL.OCR_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
        t2n_cfg.MODEL.OCR_EXTRA.STAGE4.NUM_MODULES = 3
        t2n_cfg.MODEL.OCR_EXTRA.STAGE4.NUM_BRANCHES = 4
        t2n_cfg.MODEL.OCR_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        t2n_cfg.MODEL.OCR_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]

        t2n_cfg.OPTIONS.INIT_DECODER = False
        t2n_cfg.MODEL.MSCALE_LO_SCALE = 0.5
        t2n_cfg.LOSS.OCR_ALPHA = 0.4
        t2n_cfg.LOSS.OCR_AUX_RMI = False
        t2n_cfg.LOSS.SUPERVISED_MSCALE_WT = 0
        t2n_cfg.MODEL.N_SCALES = None
        if getattr(t2n_cfg.MODEL, "BNFUNC", None) is None:
            t2n_cfg.MODEL.BNFUNC = torch.nn.BatchNorm2d
        if getattr(t2n_cfg.MODEL, "HRNET_CHECKPOINT", None) is None:
            t2n_cfg.MODEL.HRNET_CHECKPOINT = ""

        network_cls = MscaleOCR if getattr(self.config, "TILE2NET_MSCALE", True) else OCRNet
        self.tile2net_seg = network_cls(num_classes=self.num_classes, trunk='hrnetv2', criterion=None)

        hrnet_ckpt = getattr(self.config, "HRNET_CHECKPOINT", None)
        if hrnet_ckpt and os.path.isfile(hrnet_ckpt):
            print(f"[tile2net] Loading HRNet backbone weights from {hrnet_ckpt}")
            state = torch.load(hrnet_ckpt, map_location="cpu", weights_only=False)
            missing, unexpected = self.tile2net_seg.backbone.load_state_dict(state, strict=False)
            if missing:
                print(f"  missing keys: {missing[:5]}...")
            if unexpected:
                print(f"  unexpected keys: {unexpected[:5]}...")

        tile2net_ckpt = getattr(self.config, "TILE2NET_CHECKPOINT", None)
        if tile2net_ckpt and os.path.isfile(tile2net_ckpt):
            print(f"[tile2net] Loading full model weights from {tile2net_ckpt}")
            state = torch.load(tile2net_ckpt, map_location="cpu", weights_only=False)
            state = state.get("state_dict", state)
            state = {k.replace("module.", ""): v for k, v in state.items()}  # DataParallel prefix
            current_state = self.tile2net_seg.state_dict()
            compatible = {k: v for k, v in state.items()
                          if k in current_state and current_state[k].shape == v.shape}
            skipped = sorted(set(state) - set(compatible))
            if skipped:
                print(f"[tile2net] Skipping {len(skipped)} incompatible keys (e.g. {skipped[:4]})")
            self.tile2net_seg.load_state_dict(compatible, strict=False)

        if cfg_get(self.config, "FREEZE_ENCODER", False):
            for p in self.tile2net_seg.backbone.parameters():
                p.requires_grad = False

        self.encoder_output_dim = 512  # OCR mid channels

    def on_train_epoch_start(self):
        if torch.backends.mps.is_available() and cfg_get(self.config, 'MPS_FREEZE_BATCHNORM', True):
            if hasattr(self, 'decoder'):
                self._freeze_batchnorm_modules(self.decoder)

        # Optionally freeze the encoder for the first FREEZE_ENCODER_EPOCHS epochs
        # while the freshly initialised fuser/topology head stabilise. Encoder
        # parameters stay in the optimizer; AdamW skips them while they have no grad.
        freeze_epochs = int(cfg_get(self.config, 'FREEZE_ENCODER_EPOCHS', 0) or 0)
        if freeze_epochs > 0 and hasattr(self, 'encoder'):
            should_freeze = self.current_epoch < freeze_epochs
            for p in self.encoder.parameters():
                p.requires_grad = not should_freeze
            if self.current_epoch == 0 and should_freeze:
                print(f"[freeze] Encoder frozen for first {freeze_epochs} epoch(s).")
            if self.current_epoch == freeze_epochs:
                print(f"[freeze] Unfreezing encoder at epoch {self.current_epoch}.")

    # --------------------------------------------------------------- features
    def _fill_node_features(self, data_batch, image_embeddings):
        """Sample ``data_batch.x`` from ``image_embeddings`` [B,C,Hf,Wf] at ``data_batch.pos``.

        ``pos`` is in patch pixels and is rescaled to feature-map pixels per image.
        """
        B, C, Hf, Wf = image_embeddings.shape
        device, dtype = image_embeddings.device, image_embeddings.dtype
        if (getattr(data_batch, "x", None) is None or data_batch.x.numel() == 0
                or data_batch.x.shape[1] != C):
            data_batch.x = torch.zeros((data_batch.num_nodes, C), device=device, dtype=dtype)

        node_img_ids = data_batch.batch.to(device)
        pts_xy = data_batch.pos.to(device).float()
        patch = float(self.config.PATCH_SIZE)
        for img_id in torch.unique(node_img_ids):
            m = node_img_ids == img_id
            if not m.any():
                continue
            pts = pts_xy[m]
            # Guard against coordinates beyond the declared patch size.
            W_in = max(patch, pts[:, 0].max().item() + 1.0)
            H_in = max(patch, pts[:, 1].max().item() + 1.0)
            sx = (Wf - 1) / (W_in - 1)
            sy = (Hf - 1) / (H_in - 1)
            pts_fm = torch.empty_like(pts)
            pts_fm[:, 0] = pts[:, 0] * sx
            pts_fm[:, 1] = pts[:, 1] * sy
            data_batch.x[m] = self.node_sampler(
                image_embeddings[img_id:img_id + 1], pts_fm.unsqueeze(0)).squeeze(0)
        return data_batch

    def _fill_edge_features(self, data_batch, image_embeddings):
        """Set ``edge_visual_feat`` / ``edge_label_visual_feat`` by sampling along edges."""
        _, _, Hf, Wf = image_embeddings.shape
        device = image_embeddings.device
        node_img_ids = data_batch.batch.to(device)
        pts_xy = data_batch.pos.to(device).float()
        patch = float(self.config.PATCH_SIZE)
        out_dim = self.edge_visual_enc.mlp[-1].out_features

        def compute_for(edge_index):
            E = edge_index.shape[1] if edge_index is not None and edge_index.numel() > 0 else 0
            evf = image_embeddings.new_zeros((E, out_dim))
            if E == 0:
                return evf
            src, dst = edge_index
            for img_id in torch.unique(node_img_ids):
                nmask = node_img_ids == img_id
                emask = nmask[src] & nmask[dst]
                if not emask.any():
                    continue
                eidx = emask.nonzero(as_tuple=False).view(-1)
                feat = self.edge_visual_enc(image_embeddings[img_id:img_id + 1], pts_xy,
                                            edge_index[:, eidx], patch, Hf, Wf)
                evf[eidx] = feat.to(device=evf.device, dtype=evf.dtype)
            return evf

        data_batch.edge_visual_feat = compute_for(data_batch.edge_index)
        data_batch.edge_label_visual_feat = compute_for(getattr(data_batch, "edge_label_index", None))
        return data_batch

    def _segformer_forward(self, pixel_values):
        """Return (logits upsampled to the input size, encoder hidden states)."""
        hidden_states = [h.contiguous() for h in self.encoder(pixel_values).hidden_states]
        logits = self.decoder(hidden_states)
        logits = F.interpolate(logits, size=pixel_values.shape[2:], mode='bilinear', align_corners=False)
        return logits, hidden_states

    def _image_embeddings(self, hidden_states, detach=False):
        """Node-feature map from the SegFormer encoder (fused or last stage)."""
        if self.fuser is None:
            emb = hidden_states[-1]
            return emb.detach() if detach else emb
        n_stages, target_stage = FUSE_MODES[self.fuse_mode]
        feats = hidden_states[-n_stages:]
        if detach:
            feats = [f.detach() for f in feats]
        return self.fuser(feats, hidden_states[target_stage].shape[-2:])

    def _tile2net_forward_features(self, x):
        """Run Tile2Net HRNet-OCR, with its two-scale attention fusion for MscaleOCR.

        Returns segmentation logits [B,C,H,W] and the 1x OCR features [B,512,~H/4,~W/4].
        """
        if not hasattr(self.tile2net_seg, 'scale_attn'):
            _, _, hl = self.tile2net_seg.backbone(x)
            cls_out, _, ocr_feats = self.tile2net_seg.ocr(hl)
            logits = F.interpolate(cls_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            return logits, ocr_feats

        # MscaleOCR: mirrors Tile2Net's ``two_scale_forward``.
        x_lo = F.interpolate(x, scale_factor=t2n_cfg.MODEL.MSCALE_LO_SCALE,
                             mode='bilinear', align_corners=False)
        _, _, hl_lo = self.tile2net_seg.backbone(x_lo)
        cls_lo, _, ocr_lo = self.tile2net_seg.ocr(hl_lo)
        attn_lo = self.tile2net_seg.scale_attn(ocr_lo)
        lo_sz = x_lo.shape[2:]
        cls_lo = F.interpolate(cls_lo, size=lo_sz, mode='bilinear', align_corners=False)
        attn_lo = F.interpolate(attn_lo, size=lo_sz, mode='bilinear', align_corners=False)

        _, _, hl_hi = self.tile2net_seg.backbone(x)
        cls_hi, _, ocr_hi = self.tile2net_seg.ocr(hl_hi)
        hi_sz = x.shape[2:]
        cls_hi = F.interpolate(cls_hi, size=hi_sz, mode='bilinear', align_corners=False)

        p_lo = F.interpolate(attn_lo * cls_lo, size=hi_sz, mode='bilinear', align_corners=False)
        attn_up = F.interpolate(attn_lo, size=hi_sz, mode='bilinear', align_corners=False)
        return p_lo + (1 - attn_up) * cls_hi, ocr_hi

    def _sam_normalize(self, pixel_values, clamp=False):
        """ImageNet-normalised input -> SAM-normalised input."""
        rgb = _imagenet_to_rgb255(pixel_values)
        if clamp:
            rgb = rgb.clamp(0, 255)
        return (rgb - self.sam_road_model.pixel_mean) / self.sam_road_model.pixel_std

    def _sam_segmentation_logits(self, image_embeddings, out_hw):
        """Segmentation logits from SAM embeddings (SAM mask decoder or SAM-Road map decoder)."""
        if not cfg_get(self.config, 'USE_SAM_DECODER', False):
            return self.sam_road_model.map_decoder(image_embeddings)
        sparse_emb, dense_emb = self.sam_road_model.prompt_encoder(points=None, boxes=None, masks=None)
        low_res, _ = self.sam_road_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_road_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        return F.interpolate(low_res, size=out_hw, mode='bilinear', align_corners=False)

    def _pyg_to_sam_inputs(self, graph_batch, device):
        """PyG batch -> SAM-Road padded inputs (points [B,N,2], pairs [B,1,E,2], valid [B,1,E])."""
        data_list = graph_batch.to_data_list()
        B = len(data_list)
        max_points = max(d.num_nodes for d in data_list)
        max_pairs = max(d.edge_label_index.size(1) for d in data_list)
        graph_points = torch.zeros((B, max_points, 2), device=device)
        pairs = torch.zeros((B, 1, max_pairs, 2), dtype=torch.long, device=device)
        valid = torch.zeros((B, 1, max_pairs), dtype=torch.bool, device=device)
        for i, data in enumerate(data_list):
            if data.pos is not None:
                graph_points[i, :data.num_nodes, :] = data.pos
            num_edges = data.edge_label_index.size(1)
            if num_edges > 0:
                pairs[i, 0, :num_edges, :] = data.edge_label_index.t()
                valid[i, 0, :num_edges] = True
        return graph_points, pairs, valid

    # ---------------------------------------------------------------- forward
    def forward(self, pixel_values, graph_batch):
        """Return (segmentation logits [B,C,H,W], edge logits [E], edge probabilities [E])."""
        model_type = self.config.MODEL_TYPE
        if model_type == 'sam_road':
            return self._forward_sam_road(pixel_values, graph_batch)

        empty = torch.empty(0, device=pixel_values.device)
        if model_type == 'tile2net':
            logits, _ = self._segformer_forward(pixel_values)
            return logits, empty, empty

        if model_type == 'tile2net_topo':
            logits, image_embeddings = self._tile2net_forward_features(pixel_values)
        elif model_type == 'sam_topo':
            image_embeddings = self.sam_road_model.image_encoder(self._sam_normalize(pixel_values))
            logits = self._sam_segmentation_logits(image_embeddings, pixel_values.shape[2:])
        else:  # segformer
            # With FREEZE_SEGMENTATION only the topology head needs gradients.
            seg_no_grad = self._freeze_segmentation and self.training
            with torch.no_grad() if seg_no_grad else nullcontext():
                logits, hidden_states = self._segformer_forward(pixel_values)
                image_embeddings = self._image_embeddings(hidden_states, detach=self._detach_topo_features)
            if self._seg_only:
                return logits, empty, empty

        graph_batch = self._fill_node_features(graph_batch, image_embeddings)
        if self._use_edge_visual:
            graph_batch = self._fill_edge_features(graph_batch, image_embeddings)
        topo_logits, topo_scores = self.topo_net(graph_batch)

        if model_type == 'segformer' and self._use_seg_refine and topo_scores.numel() > 0:
            canvas = _rasterize_topo_canvas(
                graph_batch.pos, graph_batch.edge_label_index, topo_scores, graph_batch.batch,
                pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3])
            logits = self.seg_refine(logits, canvas)
        return logits, topo_logits, topo_scores

    def _forward_sam_road(self, pixel_values, graph_batch):
        rgb = _imagenet_to_rgb255(pixel_values).permute(0, 2, 3, 1)   # SAM-Road expects [B,H,W,C]
        graph_points, pairs, valid = self._pyg_to_sam_inputs(graph_batch, pixel_values.device)
        empty = torch.empty(0, device=pixel_values.device)

        if pairs.shape[2] == 0:
            # SAM-Road's TopoNet cannot take empty pair tensors; segment only.
            image_embeddings = self.sam_road_model.image_encoder(self._sam_normalize(pixel_values))
            return self._sam_segmentation_logits(image_embeddings, pixel_values.shape[2:]), empty, empty

        seg_logits, _, topo_logits_padded, topo_scores_padded = self.sam_road_model(
            rgb, graph_points, pairs, valid)
        logits = seg_logits.permute(0, 3, 1, 2)                         # [B, C, H, W]

        # Unpad [B, 1, max_pairs, 1] back to the flat edge order of graph_batch.
        topo_logits, topo_scores = [], []
        for i, data in enumerate(graph_batch.to_data_list()):
            num_edges = data.edge_label_index.size(1)
            if num_edges > 0:
                topo_logits.append(topo_logits_padded[i, 0, :num_edges, 0])
                topo_scores.append(topo_scores_padded[i, 0, :num_edges, 0])
        if not topo_logits:
            return logits, empty, empty
        return logits, torch.cat(topo_logits, dim=0), torch.cat(topo_scores, dim=0)

    # ----------------------------------------------------------------- losses
    @staticmethod
    def _dice_loss(seg_logits, masks, num_classes, ignore_index=-100, eps=1e-6):
        """Soft Dice over the foreground classes 1..C-1, ignoring ``ignore_index`` pixels."""
        valid = masks != ignore_index
        safe_masks = masks.clone()
        safe_masks[~valid] = 0

        valid_f = valid.unsqueeze(1).float()
        probs = F.softmax(seg_logits, dim=1) * valid_f
        target = F.one_hot(safe_masks, num_classes=num_classes).permute(0, 3, 1, 2).float() * valid_f

        probs_fg, target_fg = probs[:, 1:], target[:, 1:]
        dims = (0, 2, 3)
        intersection = (probs_fg * target_fg).sum(dim=dims)
        cardinality = probs_fg.sum(dim=dims) + target_fg.sum(dim=dims)
        return 1.0 - ((2.0 * intersection + eps) / (cardinality + eps)).mean()

    def _compute_mask_loss(self, seg_logits, masks, imgs):
        if self._freeze_segmentation:
            return imgs.new_zeros(1)

        masks = masks.long()
        num_classes = int(seg_logits.shape[1])
        # Out-of-range labels (e.g. 255) are ignored.
        invalid = (masks < 0) | (masks >= num_classes)
        if invalid.any():
            masks = masks.clone()
            masks[invalid] = -100

        if cfg_get(self.config, 'FOCAL_LOSS_SEGMENTATION', False):
            safe_masks = masks.clone()
            safe_masks[safe_masks < 0] = 0
            tgt = F.one_hot(safe_masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
            if invalid.any():
                valid = (~invalid).unsqueeze(1).float()
                focal_map = torchvision.ops.sigmoid_focal_loss(seg_logits, tgt, reduction='none')
                base_loss = (focal_map * valid).sum() / valid.sum().clamp_min(1.0)
            else:
                base_loss = self.mask_criterion(seg_logits, tgt)
        elif hasattr(self, "ce_weights") and num_classes != self.ce_weights.numel():
            # Class count differs from the configured CE weights: fall back to unweighted CE.
            base_loss = F.cross_entropy(seg_logits, masks, ignore_index=-100)
        else:
            base_loss = self.mask_criterion(seg_logits, masks)

        if self._use_dice_loss and self._dice_loss_weight > 0:
            return base_loss + self._dice_loss_weight * self._dice_loss(seg_logits, masks, num_classes)
        return base_loss

    def _has_topology(self, topo_logits, gdata):
        return (self.config.MODEL_TYPE != 'tile2net' and not self._seg_only
                and gdata.edge_label.numel() > 0 and topo_logits.numel() > 0)

    def _edge_loss(self, topo_logits, gdata):
        """Edge BCE with a fixed positive weight and extra weight on reference bridges."""
        labels = gdata.edge_label.float()
        label_edge_index = getattr(gdata, "edge_label_index", gdata.edge_index)
        bridge_w = self._bridge_weights(label_edge_index, gdata.edge_label, gdata.num_nodes)
        per_edge = self.topo_criterion(topo_logits, labels)
        return (per_edge * bridge_w.to(labels.device)).mean()

    @torch.no_grad()
    def _bridge_weights(self, edge_index, edge_label, num_nodes):
        """Per-edge weights: positive edges that are bridges get BRIDGE_EDGE_WEIGHT."""
        E = edge_index.shape[1]
        w = edge_index.new_ones(E, dtype=torch.float32)
        bw = self._bridge_edge_weight
        if E == 0 or bw == 1.0:
            return w

        pos_mask = edge_label == 1
        if pos_mask.sum() < 2:
            return w

        G = nx.Graph()
        G.add_nodes_from(range(int(num_nodes)))
        ei_np = edge_index[:, pos_mask].cpu().numpy()
        G.add_edges_from(zip(ei_np[0], ei_np[1]))
        try:
            bridge_set = set(nx.bridges(G))
        except nx.NetworkXError:
            return w

        for eidx in pos_mask.nonzero(as_tuple=False).view(-1):
            u, v = int(edge_index[0, eidx]), int(edge_index[1, eidx])
            if (u, v) in bridge_set or (v, u) in bridge_set:
                w[eidx] = bw
        return w

    def _connectivity_loss(self, topo_scores, gdata, pairs_per_graph=16):
        """Soft path-connectivity loss.

        Samples reference-connected node pairs per graph and penalises the
        negative log of the product of predicted probabilities along the
        reference shortest path between them.
        """
        edge_index = getattr(gdata, "edge_label_index", gdata.edge_index)
        if edge_index.numel() == 0:
            return topo_scores.new_zeros(1)

        edge_to_idx = {(u, v): idx for idx, (u, v) in enumerate(edge_index.t().cpu().tolist())}
        eps = 1e-7
        all_log_probs = []
        for gid in torch.unique(gdata.batch):
            nmask = gdata.batch == gid
            nodes = nmask.nonzero(as_tuple=False).view(-1).cpu().tolist()
            if len(nodes) < 2:
                continue
            src, dst = edge_index
            emask = nmask[src] & nmask[dst] & (gdata.edge_label == 1)
            if emask.sum() < 2:
                continue

            G = nx.Graph()
            G.add_nodes_from(nodes)
            pos_ei = edge_index[:, emask].cpu().numpy()
            G.add_edges_from(zip(pos_ei[0], pos_ei[1]))
            connected_nodes = [n for n in nodes if G.degree(n) > 0]
            if len(connected_nodes) < 2:
                continue

            sampled = 0
            for _ in range(pairs_per_graph * 3):
                if sampled >= pairs_per_graph:
                    break
                u, v = random.sample(connected_nodes, 2)
                try:
                    path = nx.shortest_path(G, u, v)
                except nx.NetworkXNoPath:
                    continue
                if len(path) < 2:
                    continue
                log_prob = topo_scores.new_zeros(1)
                valid = True
                for a, b in zip(path[:-1], path[1:]):
                    eidx = edge_to_idx.get((a, b), edge_to_idx.get((b, a)))
                    if eidx is None:
                        valid = False
                        break
                    log_prob = log_prob + torch.log(topo_scores[eidx] + eps)
                if valid:
                    all_log_probs.append(log_prob)
                    sampled += 1

        if not all_log_probs:
            return topo_scores.new_zeros(1)
        return -torch.stack(all_log_probs).mean()

    def _laplacian_loss(self, topo_scores, gdata):
        """Fiedler-value connectivity loss.

        Penalises ``ReLU(lambda2(L_ref) - lambda2(L_pred) + eps)`` per patch, where
        ``L_pred`` is the Laplacian of the soft predicted adjacency and ``L_ref``
        that of the reference labels. lambda2 is zero for a disconnected graph.
        """
        edge_index_all = getattr(gdata, "edge_label_index", gdata.edge_index)
        if edge_index_all.numel() == 0 or topo_scores.numel() == 0:
            return topo_scores.new_zeros(1)

        device = topo_scores.device
        eps = 1e-6

        def eigvalsh(lap):
            # eigvalsh is not implemented on MPS; the CPU copy stays differentiable.
            return torch.linalg.eigvalsh(lap.cpu() if lap.device.type == "mps" else lap)

        all_losses = []
        for gid in torch.unique(gdata.batch):
            nmask = gdata.batch == gid
            n = int(nmask.sum().item())
            if n < 4:
                continue
            node_ids = nmask.nonzero(as_tuple=False).view(-1)
            remap = {old.item(): new for new, old in enumerate(node_ids)}
            src, dst = edge_index_all
            emask = nmask[src] & nmask[dst]
            if emask.sum() < 2:
                continue

            with torch.amp.autocast(device_type=device.type, enabled=False):
                ei = edge_index_all[:, emask]
                # eigvalsh does not support bf16.
                scores = topo_scores[emask].to(device=device, dtype=torch.float32)
                labels = gdata.edge_label[emask].to(device=device, dtype=torch.float32)
                ls = torch.tensor([remap[s.item()] for s in ei[0]], device=device)
                ld = torch.tensor([remap[d.item()] for d in ei[1]], device=device)

                A_pred = torch.zeros(n, n, device=device, dtype=torch.float32)
                A_pred[ls, ld] = scores
                A_pred[ld, ls] = scores
                L_pred = torch.diag(A_pred.sum(1)) - A_pred

                with torch.no_grad():
                    A_gt = torch.zeros(n, n, device=device, dtype=torch.float32)
                    A_gt[ls, ld] = labels
                    A_gt[ld, ls] = labels
                    fiedler_gt = eigvalsh(torch.diag(A_gt.sum(1)) - A_gt)[1]

                fiedler_pred = eigvalsh(L_pred)[1]
                all_losses.append(F.relu(fiedler_gt - fiedler_pred + eps).to(device))

        if not all_losses:
            return topo_scores.new_zeros(1)
        return torch.stack(all_losses).mean()

    # ------------------------------------------------------- training/validation
    def training_step(self, batch, batch_idx):
        imgs, masks, gdata = batch["pixel_values"], batch["labels"], batch["graph_data"]
        seg_logits, topo_logits, topo_scores = self.forward(imgs, gdata)

        mask_loss = self._compute_mask_loss(seg_logits, masks, imgs)
        if self._has_topology(topo_logits, gdata):
            topo_loss = self._edge_loss(topo_logits, gdata)
        else:
            topo_loss = imgs.new_zeros(1)

        conn_loss = imgs.new_zeros(1)
        if self._conn_loss_weight > 0 and topo_scores.numel() > 0:
            conn_loss = self._connectivity_loss(topo_scores, gdata)
        lap_loss = imgs.new_zeros(1)
        if self._laplacian_loss_weight > 0 and topo_scores.numel() > 0:
            lap_loss = self._laplacian_loss(topo_scores, gdata)

        # Linearly ramp the topology loss in over the first TOPO_WARMUP_FRACTION of steps.
        topo_warmup_scale = 1.0
        if self._topo_warmup_fraction > 0:
            total_steps = self.trainer.estimated_stepping_batches
            if total_steps > 0:
                progress = self.global_step / total_steps
                topo_warmup_scale = min(1.0, progress / self._topo_warmup_fraction)

        if self._balance_loss:
            loss = (torch.exp(-self.log_vars[0]) * mask_loss + self.log_vars[0]
                    + torch.exp(-self.log_vars[1]) * topo_loss * topo_warmup_scale + self.log_vars[1])
        else:
            effective_topo_w = self._topo_loss_weight * topo_warmup_scale
            loss = self._mask_loss_weight * mask_loss + effective_topo_w * topo_loss
        loss = loss + self._conn_loss_weight * conn_loss
        loss = loss + self._laplacian_loss_weight * lap_loss

        self.log_dict({
            "train_mask_loss": mask_loss,
            "train_topo_loss": topo_loss,
            "train_conn_loss": conn_loss,
            "train_lap_loss": lap_loss,
            "train_loss": loss,
            "topo_warmup_scale": topo_warmup_scale,
        }, prog_bar=True, batch_size=imgs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks, gdata = batch["pixel_values"], batch["labels"], batch["graph_data"]
        seg_logits, topo_logits, topo_scores = self.forward(imgs, gdata)
        with torch.no_grad():
            self.val_pixel_hist += torch.bincount(masks.view(-1), minlength=self.num_classes)

        mask_loss = self._compute_mask_loss(seg_logits, masks, imgs)
        has_topology = self._has_topology(topo_logits, gdata)
        topo_loss = self._edge_loss(topo_logits, gdata) if has_topology else imgs.new_zeros(1)
        if self._balance_loss:
            loss = (torch.exp(-self.log_vars[0]) * mask_loss + self.log_vars[0]
                    + torch.exp(-self.log_vars[1]) * topo_loss + self.log_vars[1])
        else:
            loss = self._mask_loss_weight * mask_loss + self._topo_loss_weight * topo_loss

        seg_preds = torch.argmax(seg_logits, dim=1)
        self.iou_metric.update(seg_preds, masks)

        if has_topology:
            probs = topo_scores.float().view(-1)
            target = gdata.edge_label.view(-1).long()
            self.topo_f1.update(probs, target)
            self.edge_prec.update(probs, target)
            self.edge_rec.update(probs, target)
            self.edge_cm.update((probs >= 0.5).int(), target)
            self.edge_ap.update(probs, target)
            self.edge_prc.update(probs, target)

        self.log_dict({
            "val_mask_loss": mask_loss,
            "val_topo_loss": topo_loss,
            "val_loss": loss,
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=imgs.size(0))

        # Log a few qualitative samples from one validation batch.
        if batch_idx == 3 and self.logger is not None:
            n = min(4, imgs.size(0))
            imgs_np = imgs[:n].cpu().permute(0, 2, 3, 1).numpy()
            masks_np = masks[:n].cpu().numpy()
            preds_np = seg_preds[:n].cpu().numpy()
            table = [[wandb.Image(imgs_np[i]), wandb.Image(masks_np[i]), wandb.Image(preds_np[i])]
                     for i in range(n)]
            self.logger.log_table(key="val_samples", columns=['image', 'ground_truth', 'prediction'],
                                  data=table)
        return loss

    def _log_figure(self, key, fig):
        if self.logger is not None and hasattr(self.logger, "experiment") and self.global_rank == 0:
            self.logger.experiment.log({key: wandb.Image(fig)}, commit=False)
        plt.close(fig)

    def on_validation_epoch_end(self):
        # Segmentation IoU per class (background excluded from the mean).
        try:
            class_ious = self.iou_metric.compute()
        except RuntimeError:
            class_ious = None
        for c in range(1, self.num_classes):
            value = 0.0 if class_ious is None else class_ious[c].item()
            self.log(f"iou_class_{c}_{self.id2label[c]}", value,
                     prog_bar=False, on_epoch=True, sync_dist=True)
        mean_iou = 0.0 if class_ious is None else class_ious[1:].mean().item()
        self.log("val_mean_iou", mean_iou, prog_bar=True, on_epoch=True, sync_dist=True)
        self.iou_metric.reset()

        # Class pixel histogram of the validation set.
        hist = self.val_pixel_hist.clone()
        if getattr(self.trainer, "world_size", 1) > 1:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(hist, op=dist.ReduceOp.SUM)
        try:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=[self.id2label[i] for i in range(self.num_classes)],
                        y=hist.cpu().numpy(), ax=ax, palette="Blues_r")
            ax.set_title("Validation Pixel Frequency")
            ax.set_ylabel("#pixels")
            ax.set_xlabel("Class")
            ax.tick_params(axis="x", labelrotation=30)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            self._log_figure("val_pixel_hist", fig)
        except Exception:
            pass
        self.val_pixel_hist.zero_()

        # Edge classification at threshold 0.5.
        try:
            cm = self.edge_cm.compute().int().cpu().numpy()   # [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            acc = (tn + tp) / cm.sum() if cm.sum() > 0 else 0.0
            prec = self.edge_prec.compute().item()
            rec = self.edge_rec.compute().item()
            f1 = self.topo_f1.compute().item()
        except Exception:
            tn = fp = fn = tp = acc = prec = rec = f1 = 0.0
        self.log_dict({
            "edge_TN": tn, "edge_FP": fp, "edge_FN": fn, "edge_TP": tp,
            "edge_acc": acc, "edge_prec": prec, "edge_rec": rec, "edge_f1": f1,
        }, prog_bar=False, on_epoch=True, sync_dist=True)

        # Edge PR-AUC and PR curve.
        ap = None
        try:
            ap = self.edge_ap.compute().item()
        except Exception:
            pass
        self.log("edge_pr_auc", 0.0 if ap is None else ap, prog_bar=True, on_epoch=True, sync_dist=True)
        try:
            p, r, _ = self.edge_prc.compute()
            fig = plt.figure(figsize=(4, 4))
            plt.plot(r.cpu().numpy(), p.cpu().numpy(), lw=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("PR curve" if ap is None else f"PR curve (AP={ap:.3f})")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            self._log_figure("edge_pr_curve", fig)
        except Exception:
            pass

        for metric in (self.edge_cm, self.edge_prec, self.edge_rec, self.topo_f1,
                       self.edge_ap, self.edge_prc):
            metric.reset()

    # -------------------------------------------------------------- optimizer
    @staticmethod
    def _param_groups(module, lr, weight_decay=0.01):
        """AdamW groups for ``module``; biases and norm parameters get no weight decay."""
        decay, no_decay = [], []
        for name, p in module.named_parameters():
            if p.requires_grad:
                (no_decay if _no_weight_decay(name, p) else decay).append(p)
        groups = []
        if decay:
            groups.append({'params': decay, 'lr': lr, 'weight_decay': weight_decay})
        if no_decay:
            groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.0})
        return groups

    def _sam_encoder_param_groups(self, enc_lr, weight_decay=0.01):
        encoder = self.sam_road_model.image_encoder
        if not cfg_get(self.config, 'ENCODER_LORA', False):
            return self._param_groups(encoder, enc_lr)
        # Freshly initialised LoRA adapters train at the full base LR (as in
        # SAM-Road); any other trainable encoder weights use the encoder LR.
        lora_wd, lora_nwd, other_wd, other_nwd = [], [], [], []
        for name, p in encoder.named_parameters():
            if not p.requires_grad:
                continue
            is_lora = 'linear_a' in name or 'linear_b' in name
            if is_lora:
                (lora_nwd if _no_weight_decay(name, p) else lora_wd).append(p)
            else:
                (other_nwd if _no_weight_decay(name, p) else other_wd).append(p)
        base_lr = self.config.BASE_LR
        groups = []
        for params, lr, wd in ((lora_wd, base_lr, weight_decay), (lora_nwd, base_lr, 0.0),
                               (other_wd, enc_lr, weight_decay), (other_nwd, enc_lr, 0.0)):
            if params:
                groups.append({'params': params, 'lr': lr, 'weight_decay': wd})
        return groups

    def configure_optimizers(self):
        model_type = self.config.MODEL_TYPE
        base_lr = self.config.BASE_LR
        enc_lr = base_lr * cfg_get(self.config, 'ENCODER_LR_FACTOR', 0.1)
        freeze_encoder = cfg_get(self.config, 'FREEZE_ENCODER', False)

        groups = []
        if model_type in ('sam_road', 'sam_topo'):
            groups += self._sam_encoder_param_groups(enc_lr)
            for name in ('map_decoder', 'mask_decoder'):
                module = getattr(self.sam_road_model, name, None)
                if module is not None:
                    groups += self._param_groups(module, base_lr)
            if model_type == 'sam_road':
                groups += self._param_groups(self.sam_road_model.bilinear_sampler, base_lr)
                groups += self._param_groups(self.sam_road_model.topo_net, base_lr)
            else:
                groups += self._param_groups(self.topo_net, base_lr)
        elif model_type == 'tile2net':
            if not freeze_encoder:
                groups += self._param_groups(self.encoder, enc_lr)
            groups += self._param_groups(self.decoder, base_lr)
        elif model_type == 'tile2net_topo':
            if not freeze_encoder:
                groups += self._param_groups(self.tile2net_seg.backbone, enc_lr)
            groups += self._param_groups(self.tile2net_seg.ocr, base_lr)
            if hasattr(self.tile2net_seg, 'scale_attn'):
                groups += self._param_groups(self.tile2net_seg.scale_attn, base_lr)
            groups += self._param_groups(self.topo_net, base_lr)
        elif self._freeze_segmentation:
            groups += self._param_groups(self.topo_net, base_lr)
        else:  # segformer
            if not freeze_encoder:
                groups += self._param_groups(self.encoder, enc_lr)
            groups += self._param_groups(self.decoder, base_lr)
            if not self._seg_only:
                groups += self._param_groups(self.topo_net, base_lr)
                if self.fuser is not None:
                    groups += self._param_groups(self.fuser, base_lr)

        if self._node_grid > 1:
            groups += self._param_groups(self.node_sampler, base_lr)
        if hasattr(self, "edge_visual_enc"):
            groups += self._param_groups(self.edge_visual_enc, base_lr)
        if hasattr(self, "seg_refine"):
            groups += self._param_groups(self.seg_refine, base_lr)
        if self._balance_loss:
            groups.append({'params': [self.log_vars], 'lr': base_lr, 'weight_decay': 0.0})

        optim = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.999), eps=1e-8)

        # Linear warmup, then cosine or polynomial decay, stepped every batch.
        total = self.trainer.estimated_stepping_batches
        warm = int(float(cfg_get(self.config, 'WARMUP_FRACTION', 0.05)) * total)
        schedule = cfg_get(self.config, 'LR_SCHEDULE', 'poly')
        poly_exp = float(cfg_get(self.config, 'POLY_EXPONENT', 0.9))

        def lr_lambda(step):
            if step < warm:
                return step / max(1, warm)
            t, T = step - warm, max(1, total - warm)
            if schedule == 'cosine':
                return 0.5 * (1.0 + math.cos(math.pi * t / T))
            return (1 - t / float(T)) ** poly_exp

        return {'optimizer': optim,
                'lr_scheduler': {'scheduler': LambdaLR(optim, lr_lambda), 'interval': 'step'}}

    # -------------------------------------------------------------- inference
    def infer_masks_and_img_features(self, pixel_values):
        """Return (class probabilities [B,C,H,W], image embeddings [B,D,h,w])."""
        model_type = self.config.MODEL_TYPE
        if model_type == 'tile2net':
            logits, _ = self._segformer_forward(pixel_values)
            # The graph comes from Tile2Net post-processing; no embeddings are needed.
            image_embeddings = torch.zeros(pixel_values.size(0), 1, 1, 1,
                                           device=pixel_values.device, dtype=pixel_values.dtype)
        elif model_type == 'tile2net_topo':
            with torch.no_grad():
                logits, image_embeddings = self._tile2net_forward_features(pixel_values)
        elif model_type in ('sam_road', 'sam_topo'):
            image_embeddings = self.sam_road_model.image_encoder(
                self._sam_normalize(pixel_values, clamp=True))
            logits = self._sam_segmentation_logits(image_embeddings, pixel_values.shape[2:])
        else:
            logits, hidden_states = self._segformer_forward(pixel_values)
            image_embeddings = self._image_embeddings(hidden_states)
        return F.softmax(logits, dim=1), image_embeddings

    def predict_patch_topo(self, batch):
        """Score the candidate edges of a PyG batch whose ``x`` is already filled.

        Returns (logits [E], probabilities [E]).
        """
        if self.config.MODEL_TYPE == 'sam_road':
            return self._predict_patch_topo_sam_road(batch)
        if self.config.MODEL_TYPE == 'tile2net':
            raise RuntimeError("tile2net has no topology head; use "
                               "tile2net_postprocess.seg_mask_to_graph() instead.")
        return self.topo_net(batch)

    def _predict_patch_topo_sam_road(self, batch):
        """PyG batch -> padded SAM-Road TopoNet inputs -> flat [E] outputs."""
        data_list = batch.to_data_list() if hasattr(batch, 'to_data_list') else [batch]
        B = len(data_list)
        device = batch.pos.device
        max_points = max(d.num_nodes for d in data_list)
        max_pairs = max(d.edge_label_index.size(1) if d.edge_label_index.numel() > 0 else 0
                        for d in data_list)
        if max_pairs == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        D = data_list[0].x.size(1)
        points = torch.zeros((B, max_points, 2), device=device)
        point_features = torch.zeros((B, max_points, D), device=device)
        pairs = torch.zeros((B, 1, max_pairs, 2), dtype=torch.long, device=device)
        valid = torch.zeros((B, 1, max_pairs), dtype=torch.bool, device=device)
        for i, d in enumerate(data_list):
            n = d.num_nodes
            points[i, :n, :] = d.pos
            point_features[i, :n, :] = d.x[:n]
            e = d.edge_label_index.size(1)
            if e > 0:
                pairs[i, 0, :e, :] = d.edge_label_index.t()
                valid[i, 0, :e] = True

        topo_logits, topo_scores = self.sam_road_model.topo_net(points, point_features, pairs, valid)
        all_logits, all_scores = [], []
        for i, d in enumerate(data_list):
            e = d.edge_label_index.size(1)
            if e > 0:
                all_logits.append(topo_logits[i, 0, :e, 0])
                all_scores.append(topo_scores[i, 0, :e, 0])
        if all_logits:
            return torch.cat(all_logits), torch.cat(all_scores)
        return torch.empty(0, device=device), torch.empty(0, device=device)
