"""
YarnGPT training — fine-tune SmolLM2-360M on Ghanaian-English audio codes.

Reads /workspace/yarngpt_train_data.csv (produced by yarngpt_data_prep.py),
adds special tokens, resizes embeddings, and trains with standard LM loss.

Run on RunPod:
  cd /workspace
  python Ghana-tts/training/yarngpt_train.py
"""

import os
import re
import sys
import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMOLLM_PATH = "HuggingFaceTB/SmolLM2-360M"
TRAIN_CSV = "/workspace/yarngpt_train_data.csv"
OUTPUT_DIR = "/workspace/yarngpt_ghana"

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
WARMUP_STEPS = 50
SAVE_EVERY = 500  # save checkpoint every N steps
LOG_EVERY = 25

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load tokenizer + add special tokens
# ---------------------------------------------------------------------------
print("\nLoading SmolLM2-360M tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SMOLLM_PATH)

# Special tokens (exactly as YarnGPT)
special_tokens = [
    "<|im_start|>", "<|im_end|>",
    "<|text_start|>", "<|text_end|>",
    "<|audio_start|>", "<|audio_end|>",
    "<|code_start|>", "<|code_end|>",
    "<|text_sep|>",
]
tokenizer.add_tokens(special_tokens)

# Audio code tokens
audio_tokens = [f"<|{i}|>" for i in range(0, 2024)]
tokenizer.add_tokens(audio_tokens)

# Time tokens
time_tokens = [f"<|t_{round(i,2)}|>" for i in np.arange(0, 10, 0.01)]
tokenizer.add_tokens(time_tokens)

# Language tokens
tokenizer.add_tokens(["<|english|>", "<|hausa|>", "<|igbo|>", "<|yoruba|>"])

# Pad token
tokenizer.pad_token_id = 0

print(f"  Tokenizer vocab: {len(tokenizer)}")

# ---------------------------------------------------------------------------
# Load model + resize embeddings
# ---------------------------------------------------------------------------
print("\nLoading SmolLM2-360M model...")
model = AutoModelForCausalLM.from_pretrained(SMOLLM_PATH, torch_dtype="auto").to(device)
print(f"  Base params: {model.num_parameters():,}")
print(f"  Base memory: {model.get_memory_footprint() / 1e6:.1f} MB")

model.resize_token_embeddings(len(tokenizer))
print(f"  Resized embeddings to: {model.config.vocab_size}")
print(f"  New params: {model.num_parameters():,}")

model = torch.compile(model)
model.train()

# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------
print(f"\nLoading training data from {TRAIN_CSV}...")
train_data = pd.read_csv(TRAIN_CSV)
print(f"  {len(train_data)} training examples")

# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------
class YarnDataset(Dataset):
    def __init__(self, dataset):
        self.ds = dataset
        super().__init__()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        prompt = self.ds.iloc[idx]["0"]
        return tokenizer(prompt)


yarn_dataset = YarnDataset(train_data)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(
    yarn_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    shuffle=True,
)

print(f"  DataLoader: {len(dataloader)} batches (batch_size={BATCH_SIZE})")

# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)

num_training_steps = len(dataloader) * NUM_EPOCHS
num_decay_start = int(num_training_steps * 0.8)  # constant for 80%, then decay

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    elif step >= num_decay_start:
        return 1 - (step - num_decay_start) / (num_training_steps - num_decay_start)
    else:
        return 1

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"\nTraining config:")
print(f"  Epochs:          {NUM_EPOCHS}")
print(f"  Batch size:      {BATCH_SIZE}")
print(f"  Learning rate:   {LEARNING_RATE}")
print(f"  Warmup steps:    {WARMUP_STEPS}")
print(f"  Total steps:     {num_training_steps}")
print(f"  Decay starts at: {num_decay_start}")
print(f"  Save every:      {SAVE_EVERY} steps")
print(f"  Log every:       {LOG_EVERY} steps")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"STARTING TRAINING")
print(f"{'=' * 60}\n")

global_step = 0
all_losses = []

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        epoch_losses.append(loss_val)
        all_losses.append(loss_val)
        global_step += 1

        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "step": global_step,
        })

        if global_step % LOG_EVERY == 0:
            avg_loss = np.mean(all_losses[-LOG_EVERY:])
            print(f"  [step {global_step}] avg_loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if global_step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"step_{global_step}")
            print(f"  [checkpoint] Saving to {ckpt_path}...")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    # End of epoch
    avg_epoch_loss = np.mean(epoch_losses)
    print(f"\n  Epoch {epoch+1} done. Avg loss: {avg_epoch_loss:.4f}\n")

    # Save epoch checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    print(f"  Saving epoch checkpoint to {ckpt_path}...")
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)

# ---------------------------------------------------------------------------
# Save final model
# ---------------------------------------------------------------------------
final_path = os.path.join(OUTPUT_DIR, "final")
print(f"\nSaving final model to {final_path}...")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

# Save loss curve
loss_df = pd.DataFrame({"step": range(1, len(all_losses)+1), "loss": all_losses})
loss_df.to_csv(os.path.join(OUTPUT_DIR, "loss_curve.csv"), index=False)

print(f"\n{'=' * 60}")
print(f"TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"  Final loss:     {np.mean(all_losses[-100:]):.4f}")
print(f"  Total steps:    {global_step}")
print(f"  Model saved:    {final_path}")
print(f"  Loss curve:     {os.path.join(OUTPUT_DIR, 'loss_curve.csv')}")
print(f"\nTest inference:")
print(f"  python /workspace/Ghana-tts/training/yarngpt_infer.py --model_path {final_path}")
