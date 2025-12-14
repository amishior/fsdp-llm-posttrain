from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def env_default(value: Optional[str], env_key: str, fallback: Optional[str] = None) -> Optional[str]:
    if value is not None:
        return value
    return os.environ.get(env_key, fallback)
