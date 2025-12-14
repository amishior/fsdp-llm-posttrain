from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import load_dataset, get_dataset_config_names

def pick_config(dataset: str, prefer: List[str]) -> Optional[str]:
    try:
        configs = get_dataset_config_names(dataset)
    except Exception:
        return None
    # exact match first
    for p in prefer:
        for c in configs:
            if c == p:
                return c
    # contains match
    for p in prefer:
        for c in configs:
            if p.lower() in c.lower():
                return c
    return None

def stream_dataset(dataset: str, split: str = "train", config: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    if config:
        return load_dataset(dataset, name=config, split=split, streaming=True)
    return load_dataset(dataset, split=split, streaming=True)

def take(it: Iterable[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    out = []
    for i, ex in enumerate(it):
        out.append(ex)
        if i + 1 >= n:
            break
    return out
