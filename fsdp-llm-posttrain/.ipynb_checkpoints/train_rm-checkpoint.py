from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from posttrain.data.hf import stream_dataset, pick_config
from posttrain.data.format_infer import infer_preference_fields
from posttrain.train.common import make_scheduler, bench_stats
from posttrain.utils.logging import setup_logger
from posttrain.utils.misc import get_device, set_seed, save_json
from posttrain.train.common import make_optimizer

def build_pair_text(prompt: str, completion: str) -> str:
    return f"### 指令\n{prompt.strip()}\n\n### 回复\n{completion.strip()}"

def collate_rm(batch: List[Dict[str, Any]], tokenizer, seq_len: int):
    # Each item has: prompt, chosen, rejected
    chosen_text = [build_pair_text(b["prompt"], b["chosen"]) for b in batch]
    rejected_text = [build_pair_text(b["prompt"], b["rejected"]) for b in batch]
    tok_c = tokenizer(chosen_text, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
    tok_r = tokenizer(rejected_text, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
    return tok_c, tok_r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dataset", type=str, default="opencsg/UltraFeedback-chinese")
    ap.add_argument("--dataset-config", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--output-dir", type=str, default="runs/rm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stream-take", type=int, default=3000)
    args = ap.parse_args()

    logger = setup_logger(log_file=os.path.join(args.output_dir, "train.log"))
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sequence classification head for reward model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device).train()

    if args.dataset_config is None:
        args.dataset_config = pick_config(args.dataset, prefer=[
            "ultrafeedback-chinese-binarized-lowest",
            "ultrafeedback-chinese-binarized-random",
            "binarized",
            "lowest",
            "random",
        ])
        logger.info(f"Auto-picked dataset config: {args.dataset_config}")

    ds_stream = stream_dataset(args.dataset, split="train", config=args.dataset_config)
    cached = []
    for ex in ds_stream:
        try:
            prompt, chosen, rejected = infer_preference_fields(ex)
        except Exception:
            continue
        cached.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if len(cached) >= args.stream_take:
            break
    if not cached:
        raise RuntimeError("No usable preference samples for RM training. Try another dataset/config.")
    logger.info(f"Cached RM samples: {len(cached)}")

    dl = DataLoader(cached, batch_size=args.micro_batch_size, shuffle=True,
                    collate_fn=lambda b: collate_rm(b, tokenizer, args.seq_len))

    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    step = 0
    start_t = time.time()
    tokens_seen = 0
    optimizer.zero_grad(set_to_none=True)

    # Pairwise Bradley-Terry loss: -log sigmoid(r_c - r_r)
    while step < args.max_steps:
        for tok_c, tok_r in dl:
            tok_c = {k: v.to(device) for k, v in tok_c.items()}
            tok_r = {k: v.to(device) for k, v in tok_r.items()}
            tokens_seen += int(tok_c["attention_mask"].sum().item() + tok_r["attention_mask"].sum().item())

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                r_c = model(**tok_c).logits.squeeze(-1)  # [B]
                r_r = model(**tok_r).logits.squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(r_c - r_r).mean()
                loss = loss / args.grad_accum_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if args.fp16:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                logger.info(f"step={step} loss={loss.item()*args.grad_accum_steps:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

            step += 1
            if step >= args.max_steps:
                break

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    stats = bench_stats(start_t, steps=args.max_steps, tokens=tokens_seen)
    save_json(os.path.join(args.output_dir, "benchmark.json"), stats.to_dict())
    logger.info(f"Done. Saved to {args.output_dir}. Bench: {stats.to_dict()}")

if __name__ == "__main__":
    main()
