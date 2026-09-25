import json
from pathlib import Path


OUTPUT = Path("data/train-v0.2.jsonl")


CATEGORY_CONFIG = {

    "GOODS": {
        "workspace": "GOODS_NEED"
    },

    "WORKS": {
        "workspace": "WORKS_NEED"
    },

    "CONSULTANCY": {
        "workspace": "CONSULTANCY_NEED"
    },

    "NON_CONSULTANCY": {
        "workspace": "NON_CONSULTANCY_NEED"
    }
}


examples = [
    # =====================================================
    # GOODS — 10
    # =====================================================

    ("We need 120 desktop computers for our branch offices.",
     "GOODS", "Desktop computers", 120, "computers"),

    ("The laboratory needs 40 microscopes.",
     "GOODS", "Laboratory microscopes", 40, "microscopes"),

    ("We want to purchase six ambulances for regional hospitals.",
     "GOODS", "Ambulances", 6, "vehicles"),

    ("Our offices require 200 desks and 200 office chairs.",
     "GOODS", "Office furniture", 400, "items"),

    ("We need 15 diesel generators for remote facilities.",
     "GOODS", "Diesel generators", 15, "generators"),

    ("The organization needs 80 wireless access points.",
     "GOODS", "Wireless access points", 80, "access points"),

    ("We need medical refrigerators for vaccine storage.",
     "GOODS", "Medical refrigerators", None, None),

    ("Purchase protective uniforms for field employees.",
     "GOODS", "Protective uniforms", None, None),

    ("We require spare parts for our existing vehicle fleet.",
     "GOODS", "Vehicle spare parts", None, None),

    ("The department needs office stationery for the coming year.",
     "GOODS", "Office stationery", None, None),


    # =====================================================
    # WORKS — 10
    # =====================================================

    ("We need to build a new regional health center.",
     "WORKS", "Construction of regional health center", None, None),

    ("The organization plans to rehabilitate a 20 kilometer road.",
     "WORKS", "Road rehabilitation", 20, "kilometers"),

    ("We need to extend the existing headquarters building.",
     "WORKS", "Building extension", None, None),

    ("The roof of our administration building needs replacement.",
     "WORKS", "Roof replacement", None, None),

    ("We need to construct a drainage system around the facility.",
     "WORKS", "Drainage construction", None, None),

    ("We want to install a new electrical system in an existing building.",
     "WORKS", "Electrical installation works", None, None),

    ("We need to construct a perimeter fence around the compound.",
     "WORKS", "Perimeter fence construction", None, None),

    ("The school requires rehabilitation of its classrooms.",
     "WORKS", "Classroom rehabilitation", None, None),

    ("We need to construct a water distribution network.",
     "WORKS", "Water infrastructure construction", None, None),

    ("The parking area needs civil works and resurfacing.",
     "WORKS", "Parking area civil works", None, None),


    # =====================================================
    # CONSULTANCY — 10
    # =====================================================

    ("We need consultants to prepare an ICT strategy.",
     "CONSULTANCY", "ICT strategy consultancy", None, None),

    ("We need an engineering consultant to prepare the design for a bridge.",
     "CONSULTANCY", "Bridge engineering design consultancy", None, None),

    ("We require a consultant to conduct an environmental impact assessment.",
     "CONSULTANCY", "Environmental impact assessment consultancy", None, None),

    ("We need an external firm to perform a financial audit.",
     "CONSULTANCY", "Financial audit consultancy", None, None),

    ("We want experts to develop our five-year strategic plan.",
     "CONSULTANCY", "Strategic planning consultancy", None, None),

    ("We need a consultant to conduct an organizational restructuring study.",
     "CONSULTANCY", "Organizational restructuring consultancy", None, None),

    ("We require engineering supervision for a road construction project.",
     "CONSULTANCY", "Engineering supervision consultancy", None, None),

    ("We need a firm to conduct market research and analysis.",
     "CONSULTANCY", "Market research consultancy", None, None),

    ("We need legal advisors to review a complex contractual framework.",
     "CONSULTANCY", "Legal advisory consultancy", None, None),

    ("We need specialists to assess the organization's cybersecurity maturity.",
     "CONSULTANCY", "Cybersecurity assessment consultancy", None, None),


    # =====================================================
    # NON-CONSULTANCY — 10
    # =====================================================

    ("We require guards to provide security services at five branch offices.",
     "NON_CONSULTANCY", "Security guard services", None, None),

    ("We need daily janitorial services for our headquarters.",
     "NON_CONSULTANCY", "Janitorial services", None, None),

    ("We require buses to transport employees between offices.",
     "NON_CONSULTANCY", "Employee transportation services", None, None),

    ("We need a company to print and bind training manuals.",
     "NON_CONSULTANCY", "Printing and binding services", None, None),

    ("We need catering services for a three-day conference.",
     "NON_CONSULTANCY", "Catering services", None, None),

    ("We require routine maintenance services for office air conditioners.",
     "NON_CONSULTANCY", "Air conditioner maintenance services", None, None),

    ("We need a logistics company to transport equipment between regions.",
     "NON_CONSULTANCY", "Logistics and transportation services", None, None),

    ("We require pest control services at our storage facilities.",
     "NON_CONSULTANCY", "Pest control services", None, None),

    ("We need waste collection services for our facilities.",
     "NON_CONSULTANCY", "Waste collection services", None, None),

    ("We need routine grounds maintenance at our office compound.",
     "NON_CONSULTANCY", "Grounds maintenance services", None, None),
]


def build_output(
    prompt,
    category,
    description,
    quantity,
    unit,
):

    workspace = CATEGORY_CONFIG[category]["workspace"]

    return {
        "procurementCategory": category,

        "confidence": 0.95,

        "summary": description,

        "extractedInformation": {
            "description": description,
            "quantity": quantity,
            "unit": unit,
            "estimatedValue": None,
            "currency": None,
            "duration": None,
            "locations": []
        },

        "missingInformation": [
            "budget information",
            "delivery or performance location",
            "required timeframe",
            "detailed requirements"
        ],

        "recommendedWorkspace": workspace,

        "requiresRuleRetrieval": False,

        "ruleTopics": [],

        "risks": [],

        "clarificationQuestions": [
            "What budget is available?",
            "Where will the requirement be delivered or performed?",
            "What timeframe is required?",
            "What detailed requirements must be satisfied?"
        ],

        "requiresHumanReview": True
    }


records = []


for (
    prompt,
    category,
    description,
    quantity,
    unit
) in examples:

    output = build_output(
        prompt,
        category,
        description,
        quantity,
        unit
    )

    records.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
            }
        ]
    })


# =========================================================
# RULE-DEPENDENT CASES — 5
# =========================================================

rule_examples = [

    (
        "Can we use request for quotations for this purchase?",
        "GOODS",
        "procurement method rules"
    ),

    (
        "Which procurement procedure applies to a purchase valued at 12 million ETB?",
        "GOODS",
        "procurement method thresholds"
    ),

    (
        "Does this procurement require approval from the head of the organization?",
        "UNDETERMINED",
        "approval authority rules"
    ),

    (
        "What minimum bidding period must we give suppliers?",
        "UNDETERMINED",
        "minimum bidding period rules"
    ),

    (
        "Are we allowed to use direct procurement for this requirement?",
        "UNDETERMINED",
        "direct procurement rules"
    ),
]


for prompt, category, topic in rule_examples:

    workspace = (
        CATEGORY_CONFIG[category]["workspace"]
        if category in CATEGORY_CONFIG
        else "CLARIFICATION_REQUIRED"
    )

    output = {
        "procurementCategory": category,

        "confidence": (
            0.90
            if category != "UNDETERMINED"
            else 0.55
        ),

        "summary":
            "Procurement question requiring authoritative rule retrieval.",

        "extractedInformation": {
            "description": None,
            "quantity": None,
            "unit": None,
            "estimatedValue": None,
            "currency": None,
            "duration": None,
            "locations": []
        },

        "missingInformation": [
            "applicable jurisdiction",
            "procuring entity context",
            "funding source"
        ],

        "recommendedWorkspace": workspace,

        "requiresRuleRetrieval": True,

        "ruleTopics": [
            topic
        ],

        "risks": [
            {
                "type": "RULE_DEPENDENCY",
                "description":
                    "The answer requires current authoritative procurement rules."
            }
        ],

        "clarificationQuestions": [
            "Which procurement rules or jurisdiction apply?",
            "Which procuring entity is conducting the procurement?",
            "What is the funding source?"
        ],

        "requiresHumanReview": True
    }

    records.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
            }
        ]
    })


# =========================================================
# AMBIGUOUS CASES — 5
# =========================================================

ambiguous_prompts = [

    "We need someone to maintain our IT systems.",

    "We need support for our new project.",

    "We want experts and equipment for a new laboratory.",

    "We need help improving our regional offices.",

    "We need a provider to improve our information system."
]


for prompt in ambiguous_prompts:

    output = {
        "procurementCategory": "UNDETERMINED",

        "confidence": 0.50,

        "summary":
            "The procurement category cannot be determined reliably from the available information.",

        "extractedInformation": {
            "description": None,
            "quantity": None,
            "unit": None,
            "estimatedValue": None,
            "currency": None,
            "duration": None,
            "locations": []
        },

        "missingInformation": [
            "nature of the requirement",
            "expected deliverables",
            "scope of work"
        ],

        "recommendedWorkspace":
            "CLARIFICATION_REQUIRED",

        "requiresRuleRetrieval": False,

        "ruleTopics": [],

        "risks": [
            {
                "type": "AMBIGUOUS_REQUIREMENT",
                "description":
                    "The available information is insufficient to classify the procurement requirement reliably."
            }
        ],

        "clarificationQuestions": [
            "Are you procuring goods, construction works, professional intellectual services, or an operational service?",
            "What specific deliverables or outcomes are expected?",
            "What activities should the supplier perform?"
        ],

        "requiresHumanReview": True
    }

    records.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
            }
        ]
    })


with OUTPUT.open(
    "w",
    encoding="utf-8"
) as f:

    for record in records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


print(f"Created {len(records)} examples")
print(f"Saved to {OUTPUT}")
