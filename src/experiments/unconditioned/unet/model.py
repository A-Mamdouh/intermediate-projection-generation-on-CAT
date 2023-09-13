import torch
from torch import nn
from .config import ModelConfig, Activation
from src.utils.blocks import *
from typing import List, Tuple


_activation_dict = {
    Activation.LEAKY_RELU: nn.LeakyReLU,
    Activation.RELU: nn.ReLU,
    Activation.SILU: nn.SiLU,
}


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig, dropout=0.25):
        super().__init__()
        self._cfg = cfg
        self._Activation = _activation_dict[cfg.activation]
        self.dropout = dropout
        self._module_list, self._encoder_blks, self._decoder_blks = self._build()

    def _build(self) -> Tuple[nn.ModuleList, List[List[nn.Module]], List[List[nn.Module]]]:
        start_channels = self._cfg.start_channels
        dilation = self._cfg.dilation
        # Keep encoder and decoder layers in lists
        # Encoder starts with a double conv layer to convert input_channels to start channels
        encoder = [double_conv(2, start_channels, self._Activation, dropout=self.dropout)]
        # Decoder starts with the output prediction layer (1x1 conv with 1 channel. i.e. pixel-wise FC layer)
        decoder = []
        for d in range(self._cfg.depth):
            # Calculate input and output channels for the encoder layer
            in_channels = int(start_channels * (dilation**d))
            out_channels = int(in_channels * dilation)
            encoder.append(
                down(in_channels, out_channels, self._Activation, self._cfg.maxpool, dropout=self.dropout)
            )
            # Extend decoder with reversed decoder layer since the decoder list holds the reversed decoder blocks
            decoder.append(
                up(
                    out_channels,
                    in_channels,
                    self._Activation,
                    self._cfg.up_sample,
                    middle_channels=in_channels + out_channels,
                    dropout=self.dropout
                )
            )
        # Add final conv layer for prediction to final decoder layer
        decoder[0].append(nn.Conv2d(start_channels, out_channels=1, kernel_size=1))
        modules = []
        for blk in encoder:
            modules.extend(blk)
        for blk in reversed(decoder):
            modules.extend(blk)
        return nn.ModuleList(modules), encoder, list(reversed(decoder))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        for blk in self._encoder_blks:
            for layer in blk:
                x = layer(x)
            activations.append(x)
        activations.pop()
        for [up, *layers], act in zip(self._decoder_blks, reversed(activations)):
            x = up(x)
            x = torch.cat((x, act), 1)
            for layer in layers:
                x = layer(x)
        return x

    def load_ckpt(self, fname):
        state_dict = torch.load(fname)['state_dict']
        state_dict = {'.'.join(key.split('.')[1:]): value for key, value in state_dict.items()}
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    cfg = ModelConfig()
    model = Model(cfg)
    x = torch.rand((2, 2, 256, 256))
    y_hat = model(x)
    print("output_shape:", y_hat.shape)
