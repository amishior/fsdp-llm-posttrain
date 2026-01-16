from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

def infer_preference_fields(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (prompt, chosen, rejected) from a dataset row.
    Tries multiple common column conventions.
    """
    # canonical
    if all(k in example for k in ("prompt", "chosen", "rejected")):
        return str(example["prompt"]), str(example["chosen"]), str(example["rejected"])

    # instruction + chosen/rejected
    if "instruction" in example and "chosen" in example and "rejected" in example:
        return str(example["instruction"]), str(example["chosen"]), str(example["rejected"])

    # question + response_chosen/response_rejected
    if "question" in example and "response_chosen" in example and "response_rejected" in example:
        return str(example["question"]), str(example["response_chosen"]), str(example["response_rejected"])

    # query + chosen/rejected
    if "query" in example and "chosen" in example and "rejected" in example:
        return str(example["query"]), str(example["chosen"]), str(example["rejected"])

    if "instruction" in example:
        prompt = str(example["instruction"])
        for ck, rk in (("chosen", "rejected"), ("best_response", "worst_response"), ("best", "worst")):
            if ck in example and rk in example:
                return prompt, str(example[ck]), str(example[rk])

    raise KeyError(f"Cannot infer preference fields from keys={list(example.keys())[:30]}")
