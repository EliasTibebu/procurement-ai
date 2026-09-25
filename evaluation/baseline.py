import json
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/qwen3-0.6b-base"
OUTPUT_PATH = "evaluation/baseline.jsonl"

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
    }
]

SYSTEM_PROMPT = """
You are a procurement assistant.

Analyze the user's procurement request and provide a useful response.
If important information is missing, identify it.
Do not assume facts that were not provided.
""".strip()


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model.eval()

results = []

for index, case in enumerate(TEST_CASES, start=1):

    print()
    print("=" * 70)
    print(f"[{index}/{len(TEST_CASES)}] {case['id']}")
    print("=" * 70)
    print("USER:")
    print(case["prompt"])

    # Qwen3-0.6B-Base is a base model rather than an instruction-tuned
    # assistant, so keep the evaluation formatting simple and repeatable.
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User request:\n{case['prompt']}\n\n"
        f"Procurement assistant response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    start = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    elapsed = time.time() - start

    token_count = len(generated_tokens)

    tokens_per_second = (
        token_count / elapsed
        if elapsed > 0
        else 0
    )

    print()
    print("MODEL:")
    print(response)

    print()
    print(
        f"{token_count} tokens | "
        f"{elapsed:.2f}s | "
        f"{tokens_per_second:.2f} tok/s"
    )

    results.append({
        "id": case["id"],
        "prompt": case["prompt"],
        "response": response,
        "generated_tokens": token_count,
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(tokens_per_second, 2),
    })


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for result in results:
        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )


print()
print("=" * 70)
print(f"Baseline saved to {OUTPUT_PATH}")
print("=" * 70)
