#!/usr/bin/env bash
python train_dpo.py --model Qwen/Qwen3-0.6B --dataset opencsg/UltraFeedback-chinese --max-steps 50 --bf16 --gradient-checkpointing --output-dir runs/dpo_qwen3
