
import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(
    "schemas/procurement-analysis-v1.json"
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--results",
    default="evaluation/after-training.jsonl",
    help="Evaluation results JSONL"
)

args = parser.parse_args()

RESULTS_PATH = Path(args.results)


TEST_CASES_PATH = Path(
    "evaluation/test_cases.json"
)


# -------------------------------------------------------
# Load schema
# -------------------------------------------------------

with SCHEMA_PATH.open(
    encoding="utf-8"
) as f:
    schema = json.load(f)

validator = Draft202012Validator(schema)


# -------------------------------------------------------
# Load expectations
# -------------------------------------------------------

with TEST_CASES_PATH.open(
    encoding="utf-8"
) as f:
    test_cases = json.load(f)

expected = {
    case["id"]: case
    for case in test_cases
}


# -------------------------------------------------------
# ID mapping from our previous evaluator
# -------------------------------------------------------

ALIASES = {
    "goods_001": "goods_laptops",
    "consultancy_001": "consultancy_architecture",
    "non_consultancy_001": "nonconsultancy_cleaning",
    "works_001": "works_warehouse",
    "legal_safety_001": "rule_method_threshold",

    "unseen_goods_001": "goods_network_switches",
    "unseen_works_001": "works_renovation",
    "unseen_consultancy_001": "consultancy_feasibility",
    "unseen_non_consultancy_001": "nonconsultancy_security",
}


# -------------------------------------------------------
# Counters
# -------------------------------------------------------

evaluated = 0

json_valid_count = 0
schema_valid_count = 0
category_correct_count = 0
workspace_correct_count = 0
rule_correct_count = 0

rows = []


# -------------------------------------------------------
# Score results
# -------------------------------------------------------

with RESULTS_PATH.open(
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        result = json.loads(line)

        result_id = result["id"]

        mapped_id = ALIASES.get(
            result_id,
            result_id
        )

        if mapped_id not in expected:
            continue

        expectation = expected[mapped_id]

        evaluated += 1

        response = result["response"]

        json_valid = False
        schema_valid = False
        category_correct = False
        workspace_correct = False
        rule_correct = False

        parsed = None
        schema_errors = []


        # -----------------------------------------------
        # JSON
        # -----------------------------------------------

        try:

            parsed = json.loads(response)

            json_valid = True
            json_valid_count += 1

        except Exception as error:

            schema_errors.append(
                f"JSON: {error}"
            )


        # -----------------------------------------------
        # Schema
        # -----------------------------------------------

        if parsed is not None:

            errors = sorted(
                validator.iter_errors(parsed),
                key=lambda e: list(e.path),
            )

            if not errors:

                schema_valid = True
                schema_valid_count += 1

            else:

                for error in errors[:5]:

                    path = ".".join(
                        str(x)
                        for x in error.path
                    )

                    schema_errors.append(
                        f"{path or '<root>'}: "
                        f"{error.message}"
                    )


            # -------------------------------------------
            # Category
            # -------------------------------------------

            if (
                parsed.get("procurementCategory")
                ==
                expectation["expectedCategory"]
            ):

                category_correct = True
                category_correct_count += 1


            # -------------------------------------------
            # Workspace
            # -------------------------------------------

            if (
                parsed.get("recommendedWorkspace")
                ==
                expectation["expectedWorkspace"]
            ):

                workspace_correct = True
                workspace_correct_count += 1


            # -------------------------------------------
            # Rule retrieval
            # -------------------------------------------

            if (
                parsed.get("requiresRuleRetrieval")
                ==
                expectation["expectedRuleRetrieval"]
            ):

                rule_correct = True
                rule_correct_count += 1


        rows.append({
            "id": mapped_id,
            "json": json_valid,
            "schema": schema_valid,
            "category": category_correct,
            "workspace": workspace_correct,
            "rules": rule_correct,
            "errors": schema_errors,
        })


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def mark(value):
    return "✓" if value else "✗"


def percentage(value, total):

    if total == 0:
        return 0

    return round(
        (value / total) * 100,
        1
    )


# -------------------------------------------------------
# Report
# -------------------------------------------------------

print()
print("=" * 90)

print(
    f"{'Test':35}"
    f"{'JSON':8}"
    f"{'Schema':10}"
    f"{'Category':12}"
    f"{'Workspace':12}"
    f"{'Rules':8}"
)

print("=" * 90)


for row in rows:

    print(
        f"{row['id'][:34]:35}"
        f"{mark(row['json']):8}"
        f"{mark(row['schema']):10}"
        f"{mark(row['category']):12}"
        f"{mark(row['workspace']):12}"
        f"{mark(row['rules']):8}"
    )


print("=" * 90)

print()
print("MODEL SCORE")
print("-" * 50)

print(
    f"Valid JSON:          "
    f"{json_valid_count}/{evaluated} "
    f"({percentage(json_valid_count, evaluated)}%)"
)

print(
    f"Valid schema:        "
    f"{schema_valid_count}/{evaluated} "
    f"({percentage(schema_valid_count, evaluated)}%)"
)

print(
    f"Category accuracy:   "
    f"{category_correct_count}/{evaluated} "
    f"({percentage(category_correct_count, evaluated)}%)"
)

print(
    f"Workspace accuracy:  "
    f"{workspace_correct_count}/{evaluated} "
    f"({percentage(workspace_correct_count, evaluated)}%)"
)

print(
    f"Rule behavior:       "
    f"{rule_correct_count}/{evaluated} "
    f"({percentage(rule_correct_count, evaluated)}%)"
)


# -------------------------------------------------------
# Schema errors
# -------------------------------------------------------

invalid_rows = [
    row
    for row in rows
    if not row["schema"]
]


if invalid_rows:

    print()
    print("=" * 90)
    print("SCHEMA FAILURES")
    print("=" * 90)

    for row in invalid_rows:

        print()
        print(row["id"])

        for error in row["errors"]:
            print("  -", error)


print()
