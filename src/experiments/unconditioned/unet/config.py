from enum import Enum
from dataclasses import dataclass
from src.utils.base_config import BaseConfig


class Activation(Enum):
    """Enum of supported activation functions"""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SILU = "silu"


@dataclass
class ModelConfig(BaseConfig):
    # Activation function used for hidden layers
    activation: "Activation" = Activation.LEAKY_RELU
    # Depth of the model
    depth: int = 4
    # Starting channels of the first encoder layer
    start_channels: int = 128
    # Factor of channel increase between layers
    dilation: float = 2.0
    # Use upsample layer for upsampling. Uses conv_transpose if False
    up_sample: bool = False
    # Use maxpooling layer for downsampling. Uses strided convolution if False
    maxpool: bool = False

    @staticmethod
    def from_dict(raw_dict) -> "ModelConfig":
        parsed = {**raw_dict}
        if parsed.get("activation") is not None:
            parsed["activation"] = Activation(raw_dict["activation"].strip().lower())
        return ModelConfig(**parsed)
