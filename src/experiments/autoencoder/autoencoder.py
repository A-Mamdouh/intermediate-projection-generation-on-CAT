import torch
from torch import nn
from src.experiments.autoencoder.config import ModelConfig, Activation
from src.utils.blocks import *


_activation_dict = {
    Activation.LEAKY_RELU: nn.LeakyReLU,
    Activation.RELU: nn.ReLU,
    Activation.SILU: nn.SiLU,
}


class AutoEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self._cfg = cfg
        self._Activation = _activation_dict[cfg.activation]
        self._layers = self._build()

    def _build(self) -> nn.ModuleList:
        start_channels = self._cfg.start_channels
        dilation = self._cfg.dilation
        # Keep encoder and decoder layers in lists
        # Encoder starts with a double conv layer to convert input_channels to start channels
        encoder = [*double_conv(2, start_channels, self._Activation)]
        # Decoder starts with the output prediction layer (1x1 conv with 1 channel. i.e. pixel-wise FC layer)
        decoder = [nn.Conv2d(start_channels, out_channels=1, kernel_size=1)]
        for d in range(self._cfg.depth):
            # Calculate input and output channels for the encoder layer
            in_channels = int(start_channels * (dilation**d))
            out_channels = int(in_channels * dilation)
            encoder.extend(
                down(in_channels, out_channels, self._Activation, self._cfg.maxpool)
            )
            # Extend decoder with reversed decoder layer since the decoder list holds the reversed decoder blocks
            decoder.extend(
                reversed(
                    up(
                        out_channels,
                        in_channels,
                        self._Activation,
                        up_sample=self._cfg.up_sample,
                    )
                )
            )
        return nn.ModuleList((*encoder, *reversed(decoder)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return x
