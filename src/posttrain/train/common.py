from __future__ import annotations
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from posttrain.utils.misc import BenchStats

@dataclass
class TrainArgs:
    max_steps: int
    micro_batch_size: int
    grad_accum_steps: int
    lr: float
    warmup_steps: int
    weight_decay: float
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    seq_len: int
    output_dir: str
    seed: int = 42
    log_every: int = 10
    save_every: int = 200

def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in ["bias", "LayerNorm.weight", "layernorm.weight", "ln.weight"]):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr
    )

def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

def maybe_max_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    m = torch.cuda.max_memory_allocated()
    return float(m) / (1024**3)

def save_model(tokenizer, model, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)

def bench_stats(start_t: float, steps: int, tokens: int) -> BenchStats:
    sec = time.time() - start_t
    return BenchStats(steps=steps, tokens=tokens, seconds=sec, max_mem_gb=maybe_max_mem_gb())
