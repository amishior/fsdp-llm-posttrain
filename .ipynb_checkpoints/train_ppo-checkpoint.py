from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForSequenceClassification

from posttrain.data.hf import stream_dataset
from posttrain.text.chat import format_sft_example
from posttrain.utils.logging import setup_logger
from posttrain.utils.misc import set_seed, get_device

def extract_prompt_from_sft(example: Dict[str, Any]) -> str:
    # Try to recover a user prompt from instruction-style samples.
    if "instruction" in example:
        inst = str(example.get("instruction", "")).strip()
        inp = str(example.get("input", "")).strip()
        return inst if not inp else f"{inst}\n\n{inp}"
    for k in ("prompt", "question", "query"):
        if k in example:
            return str(example[k]).strip()
    # fallback: treat formatted text as prompt (not ideal, but keeps it runnable)
    try:
        txt = format_sft_example(example)
        # keep first ~200 chars as prompt
        return txt[:200]
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, required=True, help="SFT checkpoint dir (HF format)")
    ap.add_argument("--reward-model", type=str, required=True, help="Reward model checkpoint dir (HF format)")
    ap.add_argument("--prompts-dataset", type=str, default="Mxode/Chinese-Instruct")
    ap.add_argument("--prompts-dataset-config", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--output-dir", type=str, default="runs/ppo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stream-take", type=int, default=2000)
    args = ap.parse_args()

    logger = setup_logger(log_file=os.path.join(args.output_dir, "train.log"))
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.policy, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy with value head (TRL)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.policy, trust_remote_code=True)
    policy.to(device)

    # Reward model (sequence classification)
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True, trust_remote_code=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, trust_remote_code=True).to(device)
    reward_model.eval()

    config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.micro_batch_size,
        mini_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=1,
        log_with=None,
        optimize_cuda_cache=True,
        target_kl=0.1,
        max_grad_norm=1.0,
    )
    ppo_trainer = PPOTrainer(config=config, model=policy, tokenizer=tokenizer)

    # Prompt pool (streaming)
    ds_stream = stream_dataset(args.prompts_dataset, split="train", config=args.prompts_dataset_config)
    prompts = []
    for ex in ds_stream:
        p = extract_prompt_from_sft(ex)
        if p and len(p) >= 4:
            prompts.append(p)
        if len(prompts) >= args.stream_take:
            break
    if not prompts:
        raise RuntimeError("No prompts found from prompt dataset. Try a different dataset or config.")
    logger.info(f"Cached prompts: {len(prompts)}")

    os.makedirs(args.output_dir, exist_ok=True)

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

    for step in range(args.max_steps):
        prompt = prompts[step % len(prompts)]
        query_tensors = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids.to(device)

        # Generate response
        response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)
        response_text = tokenizer.decode(response_tensors[0, query_tensors.shape[1]:], skip_special_tokens=True)

        # Compute reward
        rm_inp = rm_tokenizer([f"### 指令\n{prompt}\n\n### 回复\n{response_text}"],
                              return_tensors="pt", truncation=True, max_length=args.seq_len, padding=True).to(device)
        with torch.no_grad():
            reward = reward_model(**rm_inp).logits.squeeze(-1)
        rewards = [reward.detach().cpu()]

        # PPO step
        stats = ppo_trainer.step([query_tensors[0]], [response_tensors[0, query_tensors.shape[1]:]], rewards)

        if step % 5 == 0:
            logger.info(f"step={step} reward={float(reward.item()):.4f}")

    # Save policy
    policy.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Done. Saved PPO policy to {args.output_dir}")

if __name__ == "__main__":
    main()
