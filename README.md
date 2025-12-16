# fsdp-llm-posttrain

Single-GPU **LLM post-training** (SFT / DPO / Reward Model / PPO).

- Default model: **Qwen3** .  
- Optional model: **Gemma 3** .

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) SFT
This uses `Mxode/Chinese-Instruct` (streaming) and trains for a few steps by default.
```bash
python train_sft.py --model Qwen/Qwen3-0.6B --dataset Mxode/Chinese-Instruct --max-steps 50
```

### 3) DPO
This uses `opencsg/UltraFeedback-chinese` (tries to auto-pick a binarized config).
```bash
python train_dpo.py --model Qwen/Qwen3-0.6B --dataset opencsg/UltraFeedback-chinese --max-steps 50
```

### 4) Reward Model (pairwise ranking)
```bash
python train_rm.py --base-model Qwen/Qwen3-0.6B --dataset opencsg/UltraFeedback-chinese --max-steps 100
```

### 5) PPO (TRL)
PPO needs a **policy** (SFT checkpoint) and a **reward model** checkpoint.
```bash
# (example) train a short SFT to get a policy checkpoint
python train_sft.py --model Qwen/Qwen3-0.6B --dataset Mxode/Chinese-Instruct --max-steps 200 --output-dir runs/qwen3_sft

# train reward model
python train_rm.py --base-model Qwen/Qwen3-0.6B --dataset opencsg/UltraFeedback-chinese --max-steps 200 --output-dir runs/qwen3_rm

# PPO
python train_ppo.py --policy runs/qwen3_sft --reward-model runs/qwen3_rm --prompts-dataset Mxode/Chinese-Instruct --max-steps 50
```

## Notes

- Datasets are loaded with **Hugging Face `datasets`** and we attempt to **infer columns automatically**.
- If your environment is offline / restricted, pre-download models & datasets using HF caching.

## FSDP-ready design

Training scripts call a small set of reusable modules in `src/posttrain/`:
- `posttrain.data.*` dataset adapters & format inference
- `posttrain.text.*` chat/prompt formatting
- `posttrain.train.*` common train loop utilities

Later you can add `posttrain.dist.fsdp_strategy` and swap the backend without rewriting task logic.
