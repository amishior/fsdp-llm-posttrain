from __future__ import annotations
from typing import Any, Dict, List, Optional

def format_sft_example(example: Dict[str, Any]) -> str:
    """Best-effort: turns an arbitrary instruction dataset row into a single training text."""
    # Common patterns: instruction/input/output
    if "instruction" in example and ("output" in example or "response" in example):
        inst = str(example.get("instruction", "")).strip()
        inp = str(example.get("input", "")).strip()
        out = str(example.get("output", example.get("response", ""))).strip()
        prompt = inst if not inp else f"{inst}\n\n{inp}"
        return f"### 指令\n{prompt}\n\n### 回复\n{out}"
    # ShareGPT-style: conversations/messages
    for key in ("conversations", "messages"):
        if key in example and isinstance(example[key], list):
            msgs = example[key]
            parts = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("from") or m.get("role") or "user"
                val = m.get("value") or m.get("content") or ""
                role = str(role).lower()
                if role in ("human", "user"):
                    parts.append(f"用户：{val}")
                elif role in ("gpt", "assistant"):
                    parts.append(f"助手：{val}")
                else:
                    parts.append(f"{role}：{val}")
            return "\n".join(parts).strip()
    # Fallback: concatenate string fields
    for k in ("prompt", "question", "query", "text"):
        if k in example:
            return str(example[k]).strip()
    raise KeyError(f"Cannot infer SFT text from keys={list(example.keys())[:20]}")
