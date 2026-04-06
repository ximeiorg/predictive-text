from dataclasses import dataclass


@dataclass
class ModelConfig:
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


@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 3e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    save_steps: int = 5000
    eval_steps: int = 1000
    log_steps: int = 100


@dataclass
class DataConfig:
    vocab_path: str = "data/vocab.json"
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    vocab_size: int = 8000
    val_ratio: float = 0.05
