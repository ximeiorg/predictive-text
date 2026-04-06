from dataclasses import dataclass
from typing import Dict, Type


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

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Tiny 模型 - 极致体积 (~2M 参数)

        适用场景:
        - 低端手机
        - 极致体积要求
        - 推理速度优先
        """
        return cls(
            vocab_size=8000,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            ffn_dim=256,
            max_seq_len=64,
            dropout=0.1,
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small 模型 - 小型 (~6M 参数)

        适用场景:
        - 普通手机
        - 体积和速度平衡
        - 输入法联想
        """
        return cls(
            vocab_size=8000,
            hidden_dim=256,
            num_heads=4,
            num_layers=4,
            ffn_dim=512,
            max_seq_len=64,
            dropout=0.1,
        )

    @classmethod
    def medium(cls) -> "ModelConfig":
        """Medium 模型 - 中型 (~12M 参数)

        适用场景:
        - 高端手机
        - 精度和速度平衡
        - 更好的联想效果
        """
        return cls(
            vocab_size=8000,
            hidden_dim=384,
            num_heads=6,
            num_layers=6,
            ffn_dim=1024,
            max_seq_len=64,
            dropout=0.1,
        )

    @classmethod
    def base(cls) -> "ModelConfig":
        """Base 模型 - 默认配置 (~20M 参数)

        适用场景:
        - PC/服务器部署
        - 追求精度
        - 默认配置
        """
        return cls(
            vocab_size=8000,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            ffn_dim=2048,
            max_seq_len=128,
            dropout=0.1,
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        """Large 模型 - 大型 (~40M 参数)

        适用场景:
        - 服务器部署
        - 追求精度
        - 复杂任务
        """
        return cls(
            vocab_size=8000,
            hidden_dim=768,
            num_heads=12,
            num_layers=8,
            ffn_dim=3072,
            max_seq_len=128,
            dropout=0.1,
        )

    @classmethod
    def from_name(cls, name: str) -> "ModelConfig":
        """根据名称创建配置

        Args:
            name: 配置名称 (tiny, small, medium, base, large)

        Returns:
            ModelConfig 实例
        """
        configs = {
            "tiny": cls.tiny,
            "small": cls.small,
            "medium": cls.medium,
            "base": cls.base,
            "large": cls.large,
            "default": cls.base,  # 默认使用 base
        }

        if name not in configs:
            raise ValueError(
                f"Unknown model size: {name}. Available: {list(configs.keys())}"
            )

        return configs[name]()

    def count_parameters(self) -> int:
        """计算模型参数量"""
        # Embedding 层
        embedding_params = self.vocab_size * self.hidden_dim
        position_params = self.max_seq_len * self.hidden_dim

        # 每个 Transformer 块
        # Self-Attention: 4 * hidden_dim^2 (Q, K, V, O projections)
        attn_params = 4 * self.hidden_dim * self.hidden_dim

        # FFN: hidden_dim * ffn_dim * 2 (fc1, fc2)
        ffn_params = self.hidden_dim * self.ffn_dim * 2

        # LayerNorm: 2 * hidden_dim * 2 (ln1, ln2)
        ln_params = 2 * 2 * self.hidden_dim

        # 每层总参数
        layer_params = attn_params + ffn_params + ln_params

        # 所有层
        total_layer_params = layer_params * self.num_layers

        # 最终 LayerNorm
        final_ln = 2 * self.hidden_dim

        # LM Head (权重共享，不计入额外参数)

        total = embedding_params + position_params + total_layer_params + final_ln

        return total

    def get_size_mb(self) -> float:
        """获取模型大小 (MB，float32)"""
        params = self.count_parameters()
        bytes_size = params * 4  # float32 = 4 bytes
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


# 预定义配置列表
MODEL_SIZES = {
    "tiny": {
        "params": "~2M",
        "size_mb": "~8 MB",
        "description": "极致体积，低端手机",
        "config": ModelConfig.tiny,
    },
    "small": {
        "params": "~6M",
        "size_mb": "~24 MB",
        "description": "小型模型，推荐手机",
        "config": ModelConfig.small,
    },
    "medium": {
        "params": "~12M",
        "size_mb": "~48 MB",
        "description": "中型模型，高端手机",
        "config": ModelConfig.medium,
    },
    "base": {
        "params": "~20M",
        "size_mb": "~80 MB",
        "description": "默认配置，PC/服务器",
        "config": ModelConfig.base,
    },
    "large": {
        "params": "~40M",
        "size_mb": "~160 MB",
        "description": "大型模型，追求精度",
        "config": ModelConfig.large,
    },
}


def list_model_sizes():
    """列出所有可用的模型配置"""
    print("\n可用的模型配置:")
    print("=" * 70)
    print(f"{'名称':<10} {'参数量':<10} {'大小':<12} {'说明'}")
    print("-" * 70)
    for name, info in MODEL_SIZES.items():
        default_mark = " (默认)" if name == "base" else ""
        print(
            f"{name:<10} {info['params']:<10} {info['size_mb']:<12} {info['description']}{default_mark}"
        )
    print("=" * 70)
    print("\n使用方法:")
    print("  训练: uv run src/train.py --model-size small")
    print("  推理: uv run src/inference.py --model-size small")
    print()


@dataclass
class TrainingConfig:
    """训练配置"""

    batch_size: int = 256
    learning_rate: float = 3e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    save_steps: int = 5000
    eval_steps: int = 1000
    log_steps: int = 100

    @classmethod
    def for_model_size(cls, model_size: str) -> "TrainingConfig":
        """根据模型尺寸自动调整训练配置"""
        if model_size == "tiny":
            return cls(
                batch_size=512,
                learning_rate=5e-4,
                num_epochs=15,
            )
        elif model_size == "small":
            return cls(
                batch_size=384,
                learning_rate=4e-4,
                num_epochs=12,
            )
        elif model_size == "medium":
            return cls(
                batch_size=320,
                learning_rate=3e-4,
                num_epochs=10,
            )
        elif model_size == "large":
            return cls(
                batch_size=192,
                learning_rate=2e-4,
                num_epochs=15,
            )
        else:  # base
            return cls(
                batch_size=256,
                learning_rate=3e-4,
                num_epochs=10,
            )


@dataclass
class DataConfig:
    """数据配置"""

    vocab_path: str = "data/vocab.json"
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    vocab_size: int = 8000
    val_ratio: float = 0.05
