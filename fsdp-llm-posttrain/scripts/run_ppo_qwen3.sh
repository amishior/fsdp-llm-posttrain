#!/usr/bin/env bash
python train_ppo.py --policy runs/sft_qwen3 --reward-model runs/rm_qwen3 --prompts-dataset Mxode/Chinese-Instruct --max-steps 50 --bf16 --output-dir runs/ppo_qwen3
