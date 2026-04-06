import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.num_heads

        assert self.hidden_dim % self.num_heads == 0, (
            "hidden_dim must be divisible by num_heads"
        )

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )

        return self.o_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)

        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)

        return x


class DecoderTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config.dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Dataset already provides shifted labels (input_ids[:-1], labels[1:])
            # So logits and labels should have the same length, use them directly
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=self.config.pad_token_id,
                label_smoothing=self.config.label_smoothing,
            )

        return {"logits": logits, "loss": loss}

    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        deterministic=False,
    ):
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                context = input_ids[:, -self.config.max_seq_len :]
            else:
                context = input_ids

            logits = self(context)["logits"]
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if deterministic or temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: ModelConfig = None):
    if config is None:
        config = ModelConfig()
    return DecoderTransformer(config)
