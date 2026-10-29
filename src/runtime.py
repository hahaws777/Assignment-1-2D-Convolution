"""Seven pure convolution paths and selected-layer PyNET integration.

This module provides the PyTorch C++/CUDA extension compilation binding,
model checkpoint loading, and dynamic replacement of specific Conv2d layers
in the pretrained PyNET architecture with custom CUDA convolution paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import torch
from torch import nn
from torch.utils.cpp_extension import load
from pynet import PyNET

ROOT = Path(__file__).resolve().parents[1]
METHODS: Sequence[str] = (
    "naive",
    "const",
    "tiled",
    "im2col",
    "im2col_reg",
    "implicit",
    "gridstride",
)
TARGETS: Sequence[str] = (
    "conv_l1_d1.conv_3a.conv2d",
    "conv_l0_d1.conv2d",
)

_EXTENSION_CACHE = None


def extension() -> Any:
    """Build or retrieve the cached PyTorch CUDA extension for 7 convolutions.

    Uses `torch.utils.cpp_extension.load` to compile and link the CUDA kernels
    in `src/convolution.cu` with -O3 optimization flags.

    Returns:
        Any: Compiled Python C-extension module exposing `convolution()`.
    """
    global _EXTENSION_CACHE
    if _EXTENSION_CACHE is None:
        _EXTENSION_CACHE = load(
            name="pynet_seven_convolutions",
            sources=[str(ROOT / "src/convolution.cu")],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _EXTENSION_CACHE


def load_model(checkpoint: str | Path | None = None) -> PyNET:
    """Load pretrained PyNET level 0 model weights onto CUDA device.

    Args:
        checkpoint: Optional file path to pretrained `.pth` weights.
            Defaults to `<ROOT>/checkpoints/pynet_level_0.pth`.

    Returns:
        PyNET: Pretrained PyNET neural ISP model placed on CUDA in eval mode.
    """
    checkpoint = checkpoint or (ROOT / "checkpoints/pynet_level_0.pth")
    model = PyNET(0, instance_norm=True, instance_norm_level_1=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    clean_state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=True)
    return model.cuda().eval()


class CustomConv(nn.Module):
    """Custom PyTorch module wrapping a hand-written CUDA convolution kernel.

    Replaces standard nn.Conv2d while preserving weight tensors, bias vectors,
    and stride parameters. Surrounding operations (padding, activation,
    instance normalization) remain managed by PyNET.

    Attributes:
        weight (torch.nn.Parameter): Filter weights [K, C, R, S].
        bias (torch.nn.Parameter): Output bias vector [K].
        stride (int): Spatial stride step along H and W.
        method (str): Name of custom CUDA kernel variant.
        tile (int): Shared-memory tile dimension (8, 16, or 32).
        grid_blocks (int): Explicit grid block count for gridstride method.
    """

    def __init__(
        self,
        original: nn.Conv2d,
        method: str,
        tile: int = 16,
        grid_blocks: int = 0,
    ) -> None:
        """Initialize the custom convolution layer from an existing nn.Conv2d layer.

        Args:
            original: Source nn.Conv2d layer whose weights and bias are reused.
            method: Selected convolution method (e.g., 'tiled', 'implicit').
            tile: Spatial tile size for shared-memory kernels (default: 16).
            grid_blocks: Grid block count for gridstride kernel (0 = auto).

        Raises:
            ValueError: If original Conv2d has groups > 1, dilation != 1, or padding != 0.
        """
        super().__init__()
        if original.groups != 1 or original.dilation != (1, 1) or original.padding != (0, 0):
            raise ValueError("Expected valid dense PyNET convolution with groups=1, dilation=1, padding=0")

        self.weight = original.weight
        self.bias = original.bias
        self.stride = original.stride[0]
        self.method = method
        self.tile = tile
        self.grid_blocks = grid_blocks
        self.ext = extension()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the forward pass using the custom CUDA convolution kernel.

        Args:
            x: Input feature map tensor of shape [N, C, H, W] on CUDA.

        Returns:
            torch.Tensor: Convolved output tensor of shape [N, K, P, Q].
        """
        return self.ext.convolution(
            x,
            self.weight,
            self.bias,
            self.method,
            self.stride,
            0,
            self.tile,
            self.grid_blocks,
        )


def replace_convolutions(
    model: nn.Module,
    method: str,
    targets: Sequence[str] = TARGETS,
) -> nn.Module:
    """Replace targeted Conv2d layers in PyNET with CustomConv wrappers.

    Traverses module hierarchy by dot-separated attribute path and substitutes
    standard PyTorch convolutions with our custom CUDA kernel invocations.

    Args:
        model: PyTorch neural network model (e.g., PyNET).
        method: Chosen convolution kernel variant.
        targets: Dot-delimited submodule paths to substitute.

    Returns:
        nn.Module: The modified model instance with custom convolutions.
    """
    for path in targets:
        parent_path, attribute = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        original_layer = getattr(parent, attribute)
        setattr(parent, attribute, CustomConv(original_layer, method))
    return model
