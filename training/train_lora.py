import json
import os
import time

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


MODEL_PATH = "models/qwen3-0.6b-base"
DATASET_PATH = "data/train.jsonl"
OUTPUT_DIR = "output/procurement-0.6b-lora-v0.2"

MAX_LENGTH = 256


# ---------------------------------------------------------
# CPU configuration
# ---------------------------------------------------------

torch.set_num_threads(6)
torch.set_num_interop_threads(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class ProcurementDataset(Dataset):

    def __init__(self, path):

        self.examples = []

        with open(path, encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                record = json.loads(line)

                user_message = record["messages"][0]["content"]
                assistant_message = record["messages"][1]["content"]

                # Keep formatting identical and predictable.
                prefix = (
                    "You are a procurement analysis system.\n"
                    "Return only valid JSON.\n\n"
                    f"USER:\n{user_message}\n\n"
                    "ASSISTANT:\n"
                )

                full_text = prefix + assistant_message

                prefix_tokens = tokenizer(
                    prefix,
                    add_special_tokens=True,
                )["input_ids"]

                encoded = tokenizer(
                    full_text,
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]

                labels = input_ids.clone()

                # Do not train on padding.
                labels[attention_mask == 0] = -100

                # Train only on the assistant's answer.
                prefix_length = min(
                    len(prefix_tokens),
                    MAX_LENGTH
                )

                labels[:prefix_length] = -100

                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                })


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index):
        return self.examples[index]


dataset = ProcurementDataset(DATASET_PATH)

print(f"Training examples: {len(dataset)}")


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model.config.use_cache = False


# ---------------------------------------------------------
# LoRA
# ---------------------------------------------------------

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

print()
print("Trainable parameters:")

model.print_trainable_parameters()


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=1,

    learning_rate=2e-4,

    weight_decay=0.01,

    warmup_steps=0.1,

    logging_steps=1,

    save_strategy="no",

    report_to="none",

    fp16=False,
    bf16=False,

    dataloader_num_workers=0,

    remove_unused_columns=False,

    optim="adamw_torch",

    seed=42,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)


print()
print("=" * 70)
print("Starting CPU LoRA training")
print("=" * 70)

start = time.time()

trainer.train()

elapsed = time.time() - start


# ---------------------------------------------------------
# Save adapter
# ---------------------------------------------------------

print()
print("Saving LoRA adapter...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


print()
print("=" * 70)
print("Training complete")
print(f"Time: {elapsed / 60:.2f} minutes")
print(f"Adapter: {OUTPUT_DIR}")
print("=" * 70)
