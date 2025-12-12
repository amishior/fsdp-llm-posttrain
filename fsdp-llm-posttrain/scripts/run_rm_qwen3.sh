#!/usr/bin/env bash
python train_rm.py --base-model Qwen/Qwen3-0.6B --dataset opencsg/UltraFeedback-chinese --max-steps 100 --bf16 --gradient-checkpointing --output-dir runs/rm_qwen3
