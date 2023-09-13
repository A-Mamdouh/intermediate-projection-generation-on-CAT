import torch.nn as nn
from typing import Callable, NewType, List, Optional


DefaultActivation = nn.LeakyReLU
ModuleBlock = NewType("ModuleBlock", List[nn.Module])


def conv_block(
    in_channels: int,
    out_channels: int,
    Activation: Callable,
    kernel_size: int = 3,
    stride: int = 1,
    padding: str = "same",
) -> ModuleBlock:
    assert padding.lower() in ("same", "valid")
    padding = padding.lower()
    if padding == "valid":
        padding = 0
    else:
        padding = int(kernel_size / 2)
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        Activation(inplace=True),
    ]


def convt_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
    padding: str = "valid",
) -> ModuleBlock:
    assert padding.lower() in ("same", "valid")
    padding = padding.lower()
    if padding == "valid":
        padding = 0
    else:
        padding = int(kernel_size / 2)
    return [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]


def double_conv(
    in_channels: int,
    out_channels: int,
    Activation: Callable,
    middle_channels: Optional[int] = None,
    kernel_size: int = 3,
    stride: int = 1,
    padding: str = "same",
    dropout: float=0.25
) -> ModuleBlock:
    if middle_channels is None:
        middle_channels = out_channels
    return [
        *conv_block(
            in_channels, middle_channels, Activation, kernel_size, stride, padding
        ),
        *conv_block(
            middle_channels, out_channels, Activation, kernel_size, stride, padding
        ),
        nn.Dropout2d(dropout),
    ]


def down(
    in_channels: int, out_channels: int, Activation: Callable, maxpool: bool, dropout: float=0.25
) -> ModuleBlock:
    layers = []
    if maxpool:
        layers.append(nn.Maxpool2D())
    else:
        layers.extend(conv_block(in_channels, in_channels, Activation, stride=2))
    layers.extend(double_conv(in_channels, out_channels, Activation, dropout=dropout))
    return layers


def up(
    in_channels: int,
    out_channels: int,
    Activation: Callable,
    up_sample: bool,
    middle_channels: Optional[int] = None,
    dropout=0.25
) -> ModuleBlock:
    layers = []
    if middle_channels is None:
        middle_channels = in_channels
    if up_sample:
        layers.append(nn.Upsample(scale_factor=2))
    else:
        layers.extend(convt_block(in_channels, in_channels))
    layers.extend(double_conv(middle_channels, out_channels, Activation, dropout=dropout))
    return layers
