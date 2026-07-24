"""Configuration management with YAML support."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 8000
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 2048
    max_seq_len: int = 128
    dropout: float = 0.1
    label_smoothing: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    focal_loss_gamma: float = 0.0
    use_rope: bool = False
    tail_loss_weight: float = 1.0  # >1.0 增强最后几个 token 的 loss 权重
    tail_loss_len: int = 8          # 尾部分区长度

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        # 只使用 cls 定义的字段
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def count_parameters(self) -> int:
        """计算模型参数量"""
        embedding_params = self.vocab_size * self.hidden_dim
        position_params = self.max_seq_len * self.hidden_dim
        attn_params = 4 * self.hidden_dim * self.hidden_dim
        ffn_params = self.hidden_dim * self.ffn_dim * 2
        ln_params = 2 * 2 * self.hidden_dim
        layer_params = attn_params + ffn_params + ln_params
        total_layer_params = layer_params * self.num_layers
        final_ln = 2 * self.hidden_dim
        total = embedding_params + position_params + total_layer_params + final_ln
        return total

    def get_size_mb(self) -> float:
        """获取模型大小 (MB，float32)"""
        params = self.count_parameters()
        bytes_size = params * 4
        return bytes_size / (1024 * 1024)

    def __str__(self) -> str:
        params = self.count_parameters()
        size_mb = self.get_size_mb()
        return (
            f"ModelConfig("
            f"hidden={self.hidden_dim}, "
            f"heads={self.num_heads}, "
            f"layers={self.num_layers}, "
            f"ffn={self.ffn_dim}, "
            f"params={params / 1e6:.1f}M, "
            f"size={size_mb:.1f}MB)"
        )


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 256
    learning_rate: float = 3e-4
    num_epochs: int = 15
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    weight_decay: float = 0.01
    min_lr: float = 1e-5
    eval_steps: int = 500

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class DataConfig:
    """数据配置"""
    vocab_path: str = "data/vocab.json"
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    val_ratio: float = 0.05

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """从字典创建配置"""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class ConfigManager:
    """配置管理器 - 从 YAML 加载并管理配置"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: YAML 配置文件路径，默认使用项目根目录的 config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        self._raw_config = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_model_config(self, model_size: str = "base") -> ModelConfig:
        """
        获取指定模型尺寸的配置

        继承规则: base -> model_size
        """
        # 从 base 开始
        config_data = {}
        if "base" in self._raw_config:
            config_data.update(self._raw_config["base"].get("model", {}))

        # 覆盖指定尺寸的配置
        if model_size in self._raw_config and model_size != "base":
            config_data.update(self._raw_config[model_size].get("model", {}))

        return ModelConfig.from_dict(config_data)

    def get_training_config(self, model_size: str = "base") -> TrainingConfig:
        """
        获取指定模型尺寸的训练配置

        继承规则: base -> model_size
        """
        config_data = {}
        if "base" in self._raw_config:
            config_data.update(self._raw_config["base"].get("training", {}))

        if model_size in self._raw_config and model_size != "base":
            config_data.update(self._raw_config[model_size].get("training", {}))

        return TrainingConfig.from_dict(config_data)

    def get_data_config(self, model_size: str = "base") -> DataConfig:
        """
        获取指定模型尺寸的数据配置

        继承规则: base -> model_size
        """
        config_data = {}
        if "base" in self._raw_config:
            config_data.update(self._raw_config["base"].get("data", {}))

        if model_size in self._raw_config and model_size != "base":
            config_data.update(self._raw_config[model_size].get("data", {}))

        return DataConfig.from_dict(config_data)

    def get_all_configs(self, model_size: str = "base") -> tuple:
        """获取所有配置"""
        return (
            self.get_model_config(model_size),
            self.get_training_config(model_size),
            self.get_data_config(model_size),
        )

    def list_available_sizes(self) -> list:
        """列出所有可用的模型尺寸"""
        return [k for k in self._raw_config.keys() if k != "base"]

    def get_size_info(self, model_size: str) -> Dict[str, str]:
        """获取模型尺寸信息"""
        model_config = self.get_model_config(model_size)
        params = model_config.count_parameters()
        size_mb = model_config.get_size_mb()

        descriptions = {
            "small": "推荐手机配置",
            "base": "PC/服务器",
            "large": "追求精度",
        }

        return {
            "name": model_size,
            "params": f"{params / 1e6:.1f}M",
            "size_mb": f"{size_mb:.1f}MB",
            "description": descriptions.get(model_size, ""),
        }


# 全局配置管理器实例
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_model_config(model_size: str = "base") -> ModelConfig:
    """获取模型配置"""
    return get_config_manager().get_model_config(model_size)


def get_training_config(model_size: str = "base") -> TrainingConfig:
    """获取训练配置"""
    return get_config_manager().get_training_config(model_size)


def get_data_config(model_size: str = "base") -> DataConfig:
    """获取数据配置"""
    return get_config_manager().get_data_config(model_size)


def list_model_sizes():
    """列出所有可用的模型配置"""
    cm = get_config_manager()
    sizes = cm.list_available_sizes()

    print("\n可用的模型配置:")
    print("=" * 70)
    print(f"{'名称':<10} {'参数量':<10} {'大小':<12} {'说明'}")
    print("-" * 70)

    for size in sizes:
        info = cm.get_size_info(size)
        default_mark = " (默认)" if size == "base" else ""
        print(f"{info['name']:<10} {info['params']:<10} {info['size_mb']:<12} {info['description']}{default_mark}")

    print("=" * 70)
    print("\n使用方法:")
    print("  训练: uv run src/train.py --model-size small")
    print("  推理: uv run src/inference.py --model-size small")
    print()


# 兼容旧代码的 MODEL_SIZES 字典
MODEL_SIZES = {
    "small": {"config": lambda: get_model_config("small")},
    "base": {"config": lambda: get_model_config("base")},
    "large": {"config": lambda: get_model_config("large")},
}


# 兼容旧代码的 from_name 方法
def ModelConfig_from_name(name: str) -> ModelConfig:
    """根据名称创建配置"""
    return get_model_config(name)


# 兼容旧代码的 for_model_size 方法
def TrainingConfig_for_model_size(name: str) -> TrainingConfig:
    """根据模型尺寸返回训练配置"""
    return get_training_config(name)


# 为了保持兼容性，给 dataclass 添加类方法
ModelConfig.from_name = classmethod(lambda cls, name: ModelConfig_from_name(name))
TrainingConfig.for_model_size = classmethod(lambda cls, name: TrainingConfig_for_model_size(name))
