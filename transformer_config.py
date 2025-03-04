from dataclasses import dataclass, MISSING
from datetime import datetime
from typing import Optional
from omegaconf import DictConfig, OmegaConf

# Register a custom datetime resolver
OmegaConf.register_new_resolver(
    "datetime", lambda s: f'{s}_{datetime.now().strftime("%H_%M_%S")}'
)

@dataclass
class ModelConfig:
    input_channels: int = 2  # Input channels (Real + Imaginary parts)
    residual_layers: int = 30  # Number of residual layers in WaveNet
    residual_channels: int = 64  # Channels for residual blocks
    dilation_cycle_length: int = 10  # Dilation cycle length

    # Transformer U-Net parameters
    max_channels: int = 1024  # Max channel size for U-Net scaling
    kernel_size: int = 3  # Kernel size for Conv1D
    stride: int = 2  # Stride for downsampling
    depth: int = 8  # Depth of encoder-decoder
    bottleneck_dim: int = 1024  # Bottleneck self-attention dimension
    num_heads: int = 16  # Multi-head self-attention heads
    inner_dim: int = 4096  # Inner feed-forward dimension in transformer
    bottleneck_depth: int = 24  # Number of transformer layers in bottleneck

@dataclass
class DataConfig:
    root_dir: str = MISSING
    batch_size: int = 16
    num_workers: int = 4
    train_fraction: float = 0.8

@dataclass
class DistributedConfig:
    distributed: bool = False
    world_size: int = 2

@dataclass
class TrainerConfig:
    learning_rate: float = 2e-4
    max_steps: int = 1000
    max_grad_norm: Optional[float] = None
    fp16: bool = False
    log_every: int = 50
    save_every: int = 2000
    validate_every: int = 100
from dataclasses import field
@dataclass
class Config:
    model_dir: str = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(root_dir=""))
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    #model: ModelConfig = ModelConfig()
    #data: DataConfig = DataConfig(root_dir="")
    #distributed: DistributedConfig = DistributedConfig()
    #trainer: TrainerConfig = TrainerConfig()

# Function to parse configurations from YAML or CLI
def parse_configs(cfg: DictConfig, cli_cfg: Optional[DictConfig] = None) -> DictConfig:
    base_cfg = OmegaConf.structured(Config)
    merged_cfg = OmegaConf.merge(base_cfg, cfg)
    if cli_cfg is not None:
        merged_cfg = OmegaConf.merge(merged_cfg, cli_cfg)
    return merged_cfg

if __name__ == "__main__":
    base_config = OmegaConf.structured(Config)
    config = OmegaConf.load("configs/short_ofdm.yaml")
    config = OmegaConf.merge(base_config, OmegaConf.from_cli(), config)
    config = Config(**config)

    print(OmegaConf.to_yaml(config))