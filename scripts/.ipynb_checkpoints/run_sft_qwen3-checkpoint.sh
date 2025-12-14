#!/usr/bin/env bash
python train_sft.py --model Qwen/Qwen3-0.6B --dataset Mxode/Chinese-Instruct --max-steps 50 --bf16 --gradient-checkpointing --output-dir runs/sft_qwen3
