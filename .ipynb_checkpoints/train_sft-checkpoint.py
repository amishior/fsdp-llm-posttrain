from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain.data.hf import stream_dataset, take
from posttrain.text.chat import format_sft_example
from posttrain.train.common import TrainArgs, make_optimizer, make_scheduler, save_model, bench_stats
from posttrain.utils.logging import setup_logger
from posttrain.utils.misc import get_device, set_seed, save_json

def collate_sft(batch: List[Dict[str, Any]], tokenizer, seq_len: int):
    texts = [b["text"] for b in batch]
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
    tok["labels"] = tok["input_ids"].clone()
    return tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config path (optional)")
    ap.add_argument("--model", type=str, default="/home/amishor/Qwen3-0.6B")
    ap.add_argument("--dataset", type=str, default="Mxode/Chinese-Instruct")
    ap.add_argument("--dataset-config", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--output-dir", type=str, default="runs/sft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stream-take", type=int, default=5000, help="How many streamed rows to cache into memory")
    args = ap.parse_args()

    logger = setup_logger(log_file=os.path.join(args.output_dir, "train.log"))
    set_seed(args.seed)

    device = get_device()
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    # Load dataset (streaming) and cache a small subset to keep it runnable.
    ds_stream = stream_dataset(args.dataset, split="train", config=args.dataset_config)
    cached = []
    for ex in ds_stream:
        try:
            text = format_sft_example(ex)
        except Exception:
            continue
        cached.append({"text": text})
        if len(cached) >= args.stream_take:
            break
    if not cached:
        raise RuntimeError("No usable SFT samples after parsing. Try another dataset/config or inspect column names.")
    logger.info(f"Cached SFT samples: {len(cached)}")

    dl = DataLoader(cached, batch_size=args.micro_batch_size, shuffle=True,
                    collate_fn=lambda b: collate_sft(b, tokenizer, args.seq_len))

    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    step = 0
    start_t = time.time()
    tokens_seen = 0

    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            tokens_seen += int(batch["attention_mask"].sum().item())

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out = model(**batch)
                loss = out.loss / args.grad_accum_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                logger.info(f"step={step} loss={loss.item()*args.grad_accum_steps:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

            step += 1
            if step >= args.max_steps:
                break

    # Save
    save_model(tokenizer, model, args.output_dir)
    stats = bench_stats(start_t, steps=args.max_steps, tokens=tokens_seen)
    save_json(os.path.join(args.output_dir, "benchmark.json"), stats.to_dict())
    logger.info(f"Done. Saved to {args.output_dir}. Bench: {stats.to_dict()}")

if __name__ == "__main__":
    main()
