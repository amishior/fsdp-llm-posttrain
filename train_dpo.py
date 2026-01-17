from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain.data.hf import stream_dataset, take, pick_config
from posttrain.data.format_infer import infer_preference_fields
from posttrain.train.common import make_optimizer, make_scheduler, save_model, bench_stats
from posttrain.utils.logging import setup_logger
from posttrain.utils.misc import get_device, set_seed, save_json

def build_prompt(prompt: str) -> str:
    return f"### 指令\n{prompt.strip()}\n\n### 回复\n"

def logprob_of_completion(tokenizer, model, prompt: str, completion: str, device, seq_len: int) -> torch.Tensor:
    # Compute log p(completion | prompt) summed over completion tokens
    full = prompt + completion
    tok_full = tokenizer(full, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
    tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).to(device)

    with torch.no_grad():
        out = model(**tok_full)
        logits = out.logits  # [1, T, V]
    # Shift for next-token prediction
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = tok_full["input_ids"][:, 1:]  # [1, T-1]

    # Determine which positions correspond to completion (exclude prompt tokens)
    prompt_len = tok_prompt["input_ids"].shape[1]
    # In shifted space, completion starts at index (prompt_len-1)
    start = max(prompt_len - 1, 0)
    lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    lp_comp = lp[:, start:]
    return lp_comp.sum(dim=-1)  # [1]

def dpo_loss(pi_logp_c, pi_logp_r, ref_logp_c, ref_logp_r, beta: float) -> torch.Tensor:
    # DPO loss: -log sigma(beta*( (pi_c - pi_r) - (ref_c - ref_r) ))
    pi_diff = pi_logp_c - pi_logp_r
    ref_diff = ref_logp_c - ref_logp_r
    logits = beta * (pi_diff - ref_diff)
    return -torch.nn.functional.logsigmoid(logits)

def collate_pref(batch: List[Dict[str, Any]]):
    return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dataset", type=str, default="opencsg/UltraFeedback-chinese")
    ap.add_argument("--dataset-config", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--output-dir", type=str, default="runs/dpo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stream-take", type=int, default=2000)
    args = ap.parse_args()

    logger = setup_logger(log_file=os.path.join(args.output_dir, "train.log"))
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy + reference (frozen) start from same weights
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        trust_remote_code=True,
    )
    ref = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        trust_remote_code=True,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.to(device).train()
    ref.to(device)

    # Pick a preference-friendly config if not provided
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
        raise RuntimeError("No usable preference samples after parsing. Try another dataset/config or inspect column names.")
    logger.info(f"Cached preference samples: {len(cached)}")

    dl = DataLoader(cached, batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate_pref)

    optimizer = make_optimizer(policy, args.lr, args.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    step = 0
    start_t = time.time()
    tokens_seen = 0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch in dl:
            # micro-batch loop: we compute per-example loss and mean it
            losses = []
            for ex in batch:
                ptxt = build_prompt(ex["prompt"])
                chosen = ex["chosen"]
                rejected = ex["rejected"]

                # tokens accounting (rough)
                tokens_seen += len(tokenizer(ptxt + chosen).input_ids)
                tokens_seen += len(tokenizer(ptxt + rejected).input_ids)

                with torch.no_grad():
                    ref_logp_c = logprob_of_completion(tokenizer, ref, ptxt, chosen, device, args.seq_len)
                    ref_logp_r = logprob_of_completion(tokenizer, ref, ptxt, rejected, device, args.seq_len)

                # policy logprobs (need grad)
                full_c = ptxt + chosen
                full_r = ptxt + rejected
                tok_c = tokenizer(full_c, return_tensors="pt", truncation=True, max_length=args.seq_len).to(device)
                tok_r = tokenizer(full_r, return_tensors="pt", truncation=True, max_length=args.seq_len).to(device)
                tok_p = tokenizer(ptxt, return_tensors="pt", truncation=True, max_length=args.seq_len).to(device)
                prompt_len = tok_p["input_ids"].shape[1]
                start = max(prompt_len - 1, 0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    out_c = policy(**tok_c)
                    out_r = policy(**tok_r)
                    lp_c = torch.log_softmax(out_c.logits[:, :-1, :], dim=-1).gather(-1, tok_c["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
                    lp_r = torch.log_softmax(out_r.logits[:, :-1, :], dim=-1).gather(-1, tok_r["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
                    pi_logp_c = lp_c[:, start:].sum(dim=-1)
                    pi_logp_r = lp_r[:, start:].sum(dim=-1)

                    loss = dpo_loss(pi_logp_c, pi_logp_r, ref_logp_c, ref_logp_r, beta=args.beta).mean()
                    loss = loss / args.grad_accum_steps
                losses.append(loss)

            loss_mb = torch.stack(losses).mean()

            if args.fp16:
                scaler.scale(loss_mb).backward()
            else:
                loss_mb.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if args.fp16:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                logger.info(f"step={step} loss={loss_mb.item()*args.grad_accum_steps:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

            step += 1
            if step >= args.max_steps:
                break

    save_model(tokenizer, policy, args.output_dir)
    stats = bench_stats(start_t, steps=args.max_steps, tokens=tokens_seen)
    save_json(os.path.join(args.output_dir, "benchmark.json"), stats.to_dict())
    logger.info(f"Done. Saved to {args.output_dir}. Bench: {stats.to_dict()}")

if __name__ == "__main__":
    main()
