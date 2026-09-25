import json
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "models/qwen3-0.6b-base"
ADAPTER = "output/procurement-0.6b-lora-v0.1"
OUTPUT = "evaluation/after-training.jsonl"

torch.set_num_threads(6)
torch.set_num_interop_threads(1)


TEST_CASES = [
    {
        "id": "goods_001",
        "prompt": "We need to procure 300 laptops for our regional offices."
    },
    {
        "id": "consultancy_001",
        "prompt": "We need a consulting firm to design an enterprise architecture for our organization."
    },
    {
        "id": "non_consultancy_001",
        "prompt": "We need a company to clean our offices for one year."
    },
    {
        "id": "works_001",
        "prompt": "We need to construct a new warehouse."
    },
    {
        "id": "missing_info_001",
        "prompt": "Prepare a purchase requirement for 50 printers."
    },
    {
        "id": "legal_safety_001",
        "prompt": "Which procurement method should I use for goods worth 8 million ETB?"
    },

    # IMPORTANT:
    # These were NOT in the training dataset.
    {
        "id": "unseen_goods_001",
        "prompt": "Our organization needs 25 network switches for the new data center."
    },
    {
        "id": "unseen_works_001",
        "prompt": "We want to renovate three government office buildings."
    },
    {
        "id": "unseen_consultancy_001",
        "prompt": "We need an independent consultant to conduct a feasibility study for a new information system."
    },
    {
        "id": "unseen_non_consultancy_001",
        "prompt": "We need security guard services for our headquarters for two years."
    }
]


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)


print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)


print("Loading procurement LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
)

model.eval()


results = []


for index, case in enumerate(TEST_CASES, start=1):

    print()
    print("=" * 70)
    print(f"[{index}/{len(TEST_CASES)}] {case['id']}")
    print("=" * 70)

    prefix = (
        "You are a procurement analysis system.\n"
        "Return only valid JSON.\n\n"
        f"USER:\n{case['prompt']}\n\n"
        "ASSISTANT:\n"
    )

    inputs = tokenizer(
        prefix,
        return_tensors="pt",
    )

    start = time.time()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    elapsed = time.time() - start

    token_count = len(generated)

    print("USER:")
    print(case["prompt"])

    print()
    print("MODEL:")
    print(response)

    print()

    # Check whether it produced parseable JSON.
    valid_json = False

    try:
        parsed = json.loads(response)
        valid_json = True
        print("JSON: ✓ VALID")

    except json.JSONDecodeError as error:
        print("JSON: ✗ INVALID")
        print("Error:", error)

    print(
        f"{token_count} tokens | "
        f"{elapsed:.2f}s | "
        f"{token_count / elapsed:.2f} tok/s"
    )

    results.append({
        "id": case["id"],
        "prompt": case["prompt"],
        "response": response,
        "valid_json": valid_json,
        "tokens": token_count,
        "seconds": round(elapsed, 2),
    })


with open(
    OUTPUT,
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
print(f"Results saved: {OUTPUT}")
print("=" * 70)
