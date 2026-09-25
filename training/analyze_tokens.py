import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL = "models/qwen3-0.6b-base"
DATASET = Path("data/train.jsonl")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)


lengths = []


with DATASET.open(encoding="utf-8") as f:

    for index, line in enumerate(f, start=1):

        example = json.loads(line)

        text = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in example["messages"]
        )

        tokens = tokenizer(
            text,
            add_special_tokens=True
        )["input_ids"]

        length = len(tokens)
        lengths.append(length)

        print(
            f"Example {index}: {length} tokens"
        )


print()
print("Examples :", len(lengths))
print("Minimum  :", min(lengths))
print("Maximum  :", max(lengths))
print(
    "Average  :",
    round(sum(lengths) / len(lengths), 1)
)
