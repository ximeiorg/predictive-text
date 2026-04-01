from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 5120
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 512
    max_seq_len: int = 32
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3


@dataclass
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 5e-4
    num_epochs: int = 3
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    save_steps: int = 5000
    eval_steps: int = 1000
    log_steps: int = 100


@dataclass
class DataConfig:
    raw_data_path: str = "大王饶命.txt"
    vocab_path: str = "data/vocab.model"
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    vocab_size: int = 5120
    max_samples: int = 100_000_000
    val_ratio: float = 0.05