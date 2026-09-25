import json
from pathlib import Path

from jsonschema import Draft202012Validator


DATASET = Path("data/train.jsonl")
SCHEMA_FILE = Path("schemas/procurement-analysis-v1.json")


with SCHEMA_FILE.open(encoding="utf-8") as f:
    schema = json.load(f)

validator = Draft202012Validator(schema)

valid = 0
invalid = 0


with DATASET.open(encoding="utf-8") as f:

    for line_number, line in enumerate(f, start=1):

        if not line.strip():
            continue

        try:
            training_example = json.loads(line)

            messages = training_example["messages"]

            if len(messages) != 2:
                raise ValueError(
                    "Expected exactly one user and one assistant message"
                )

            if messages[0]["role"] != "user":
                raise ValueError("First message must be user")

            if messages[1]["role"] != "assistant":
                raise ValueError("Second message must be assistant")

            assistant_output = json.loads(
                messages[1]["content"]
            )

            errors = sorted(
                validator.iter_errors(assistant_output),
                key=lambda e: list(e.path),
            )

            if errors:

                invalid += 1

                print(f"\n❌ Line {line_number}")

                for error in errors:
                    path = ".".join(
                        str(part)
                        for part in error.path
                    )

                    print(
                        f"   {path or '<root>'}: "
                        f"{error.message}"
                    )

            else:

                valid += 1
                print(f"✓ Line {line_number}")

        except Exception as error:

            invalid += 1

            print(
                f"\n❌ Line {line_number}: {error}"
            )


print()
print("=" * 50)
print(f"Valid:   {valid}")
print(f"Invalid: {invalid}")
print("=" * 50)

if invalid:
    raise SystemExit(1)
