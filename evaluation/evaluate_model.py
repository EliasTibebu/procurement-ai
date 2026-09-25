import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "models/qwen3-0.6b-base"
TEST_CASES_FILE = "evaluation/test_cases.json"

torch.set_num_threads(6)
torch.set_num_interop_threads(1)


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--adapter",
    required=True,
    help="Path to LoRA adapter"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output JSONL file"
)

parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=300
)

args = parser.parse_args()


adapter_path = Path(args.adapter)
output_path = Path(args.output)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Load test cases
# ---------------------------------------------------------

with open(
    TEST_CASES_FILE,
    encoding="utf-8"
) as f:
    test_cases = json.load(f)


print(f"Test cases: {len(test_cases)}")
print(f"Adapter:    {adapter_path}")
print(f"Output:     {output_path}")


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------
# Base model
# ---------------------------------------------------------

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)


# ---------------------------------------------------------
# LoRA
# ---------------------------------------------------------

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    str(adapter_path),
)

model.eval()


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

results = []


for index, case in enumerate(
    test_cases,
    start=1
):

    case_id = case["id"]
    user_prompt = case["prompt"]

    print()
    print("=" * 70)
    print(
        f"[{index}/{len(test_cases)}] "
        f"{case_id}"
    )
    print("=" * 70)

    prefix = (
        "You are a procurement analysis system.\n"
        "Return only valid JSON.\n\n"
        f"USER:\n{user_prompt}\n\n"
        "ASSISTANT:\n"
    )

    inputs = tokenizer(
        prefix,
        return_tensors="pt"
    )

    start = time.time()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    elapsed = time.time() - start

    token_count = len(generated)

    tokens_per_second = (
        token_count / elapsed
        if elapsed > 0
        else 0
    )

    print("USER:")
    print(user_prompt)

    print()
    print("MODEL:")
    print(response)

    print()

    valid_json = False

    try:

        json.loads(response)

        valid_json = True

        print("JSON: ✓ VALID")

    except json.JSONDecodeError as error:

        print("JSON: ✗ INVALID")
        print("Error:", error)

    print(
        f"{token_count} tokens | "
        f"{elapsed:.2f}s | "
        f"{tokens_per_second:.2f} tok/s"
    )

    results.append({
        "id": case_id,
        "prompt": user_prompt,
        "response": response,
        "valid_json": valid_json,
        "tokens": token_count,
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(
            tokens_per_second,
            2
        )
    })


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

with output_path.open(
    "w",
    encoding="utf-8"
) as f:

    for result in results:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            )
            + "\n"
        )


print()
print("=" * 70)
print("Evaluation complete")
print(f"Results saved: {output_path}")
print("=" * 70)
