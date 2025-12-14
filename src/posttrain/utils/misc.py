from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

@dataclass
class BenchStats:
    steps: int
    tokens: int
    seconds: float
    max_mem_gb: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
