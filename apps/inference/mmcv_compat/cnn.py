"""Pure PyTorch replacements for mmcv.cnn functions.

Used by FakeSTormer's Swin Transformer and head modules.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Layer builders ───────────────────────────────────────────────────────────
# mmcv uses config dicts to select layer types. FakeSTormer only uses
# Conv2d, BatchNorm2d/LayerNorm, and Upsample, so we just ignore the
# config dict and instantiate the default.


def build_conv_layer(cfg, *args, **kwargs):
    """Build a convolution layer from config.

    FakeSTormer always uses standard Conv2d.
    cfg is ignored -- it typically specifies {'type': 'Conv2d'}.
    """
    return nn.Conv2d(*args, **kwargs)


def build_norm_layer(cfg, num_features, postfix=""):
    """Build a normalization layer from config.

    Args:
        cfg: Dict with 'type' key. Supported: 'BN', 'LN', 'GN'.
        num_features: Number of features/channels.
        postfix: Name postfix for the layer.

    Returns:
        Tuple of (name, layer) matching mmcv convention.
    """
    if cfg is None:
        cfg = {"type": "BN"}

    norm_type = cfg.get("type", "BN") if isinstance(cfg, dict) else "BN"
    name = f"norm{postfix}" if postfix else "norm"

    if norm_type in ("BN", "BN2d"):
        layer = nn.BatchNorm2d(num_features)
    elif norm_type == "LN":
        layer = nn.LayerNorm(num_features)
    elif norm_type == "GN":
        num_groups = cfg.get("num_groups", 32)
        layer = nn.GroupNorm(num_groups, num_features)
    elif norm_type == "BN3d":
        layer = nn.BatchNorm3d(num_features)
    else:
        layer = nn.BatchNorm2d(num_features)

    return name, layer


def build_upsample_layer(cfg, *args, **kwargs):
    """Build an upsample layer from config.

    FakeSTormer uses nn.ConvTranspose2d for upsampling.
    """
    if cfg is None:
        return nn.Upsample(*args, **kwargs)

    upsample_type = cfg.get("type", "nearest") if isinstance(cfg, dict) else "nearest"

    if upsample_type == "deconv":
        return nn.ConvTranspose2d(*args, **kwargs)
    return nn.Upsample(*args, **kwargs)


# ── Weight initialization ────────────────────────────────────────────────────


def constant_init(module, val, bias=0):
    """Initialize module weights to a constant value."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=0.01, bias=0):
    """Initialize module weights with normal distribution."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module, mean=0, std=0.02, a=-2, b=2, bias=0):
    """Initialize module weights with truncated normal distribution."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_(tensor, mean=0.0, std=0.02, a=-2.0, b=2.0):
    """Fill tensor with truncated normal distribution (in-place)."""
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# ── Transformer building blocks ──────────────────────────────────────────────


def build_dropout(cfg, default_args=None):
    """Build dropout layer from config.

    Args:
        cfg: Dict with 'type' key and drop_prob.

    Returns:
        Dropout module or Identity if drop_prob is 0.
    """
    if cfg is None:
        return nn.Identity()

    drop_type = cfg.get("type", "Dropout") if isinstance(cfg, dict) else "Dropout"
    drop_prob = cfg.get("drop_prob", 0.0) if isinstance(cfg, dict) else 0.0

    if drop_prob == 0.0:
        return nn.Identity()

    if drop_type == "DropPath":
        return DropPath(drop_prob)
    return nn.Dropout(p=drop_prob)


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor


class FFN(nn.Module):
    """Feed-Forward Network used in Transformers.

    Replacement for mmcv.cnn.bricks.transformer.FFN.
    """

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=None,
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.add_identity = add_identity

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        if dropout_layer is not None:
            self.dropout_layer = build_dropout(dropout_layer)
        else:
            self.dropout_layer = nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
