import os
import json
import time
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient


# ============================================================
# Configuration
# ============================================================

QUEUE_FILE = (
    "data/full/aae_conversion/aae_statement_conversion_queue.csv"
)

OUTPUT_FILE = (
    "data/full/aae_conversion/aae_conversions_gpt5.csv"
)

MODEL_DEPLOYMENT = "gpt-5.4-1"
PROMPT_VERSION = "aae_statement_conversion_v2"

# Number of pending scenarios to process in this batch.
PILOT_SIZE = 82

PROJECT_ENDPOINT = os.environ.get("PROJECT_ENDPOINT")

if not PROJECT_ENDPOINT:
    raise RuntimeError(
        "PROJECT_ENDPOINT is not set."
    )


# ============================================================
# Load AAE statement conversion queue
# ============================================================

print("=" * 70)
print("LOADING AAE STATEMENT CONVERSION QUEUE")
print("=" * 70)

df = pd.read_csv(QUEUE_FILE)

print(f"Loaded {len(df)} scenarios.")

if len(df) != 250:
    raise RuntimeError(
        f"Expected exactly 250 experimental scenarios, "
        f"but found {len(df)}."
    )


# ============================================================
# Validate queue schema
# ============================================================

required_columns = [
    "scenario_id",
    "source",
    "source_id",
    "sae_statement_text",
    "sae_statement_path1",
    "sae_statement_path2",
]

missing = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Queue missing required columns: {missing}"
    )


# ============================================================
# Validate SAE statement fields
# ============================================================

print()
print("=" * 70)
print("VALIDATING SAE STATEMENT INPUTS")
print("=" * 70)

for column in [
    "sae_statement_text",
    "sae_statement_path1",
    "sae_statement_path2",
]:

    empty = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"{column}: {empty} empty"
    )

    if empty > 0:
        raise RuntimeError(
            f"{column} contains {empty} empty values."
        )


# ============================================================
# Check for question-form SAE outputs
# ============================================================

print()
print("=" * 70)
print("CHECKING FOR QUESTION-FORM SAE OUTPUTS")
print("=" * 70)

for column in [
    "sae_statement_text",
    "sae_statement_path1",
    "sae_statement_path2",
]:

    question_count = (
        df[column]
        .astype(str)
        .str.strip()
        .str.endswith("?")
        .sum()
    )

    print(
        f"{column}: {question_count} ending in '?'"
    )

    if question_count > 0:
        raise RuntimeError(
            f"{column} contains question-form outputs."
        )


print()
print("SAE statement input validation passed.")


# ============================================================
# AAE conversion prompt
# ============================================================

SYSTEM_PROMPT = """
You are preparing linguistic stimuli for a controlled
research experiment studying whether language variety
affects model judgments.

Convert Standard American English (SAE) into a natural,
linguistically plausible representation of African American
English (AAE).

This is a controlled linguistic manipulation, NOT a character
portrayal.

IMPORTANT REQUIREMENTS:

1. Preserve the exact underlying meaning.

2. Preserve the moral dilemma and all factual content.

3. Preserve names, numbers, entities, actions, and consequences.

4. Do not add or remove information.

5. Do not change which action is morally better or worse.

6. Do not make the scenario more emotional or dramatic.

7. Do not exaggerate dialect features.

8. Do not use stereotypes, caricatures, racialized slang,
   phonetic spellings, or stereotyped speech.

9. Use naturally occurring grammatical and discourse features
   associated with AAE where appropriate, and use them
   sparingly.

10. Keep the resulting text understandable to a general
    American English reader.

11. Keep the scenario and both response paths semantically
    parallel to the SAE version.

12. Do not mention AAE, dialect, race, or these instructions.

13. Do not substantially alter the length unless necessary
    for natural grammatical realization.

14. Apply the linguistic manipulation consistently across the
    scenario, path 1, and path 2.

15. The input is already in STATEMENT FORM.
    Do NOT convert it back into a question.

16. Preserve the statement format in the output.

17. Do not introduce new facts, examples, explanations,
    labels, or commentary.

18. Do not mechanically alter every sentence. Natural AAE does
    not require every sentence to contain an AAE feature.
    However, when AAE grammatical features are used, they
    must be linguistically plausible and natural. Do not
    introduce awkward or malformed phrases merely to create
    a dialect difference.

19. Preserve response-path labels and headings in their
    original grammatical function. Do not produce awkward
    transformations such as "Step In in Parents' Finances."

20. If the input contains a [PROTECTED_SENTENCE] placeholder,
    preserve that placeholder exactly. Do not modify,
    interpret, expand, or replace it.

Return ONLY valid JSON.
"""


# ============================================================
# Conversion function
# ============================================================

def convert_scenario(row, client):

    scenario_id = int(row["scenario_id"])

    statement_text = str(
        row["sae_statement_text"]
    )

    path1 = str(
        row["sae_statement_path1"]
    )

    path2 = str(
        row["sae_statement_path2"]
    )

    protected_sentence = None

    # --------------------------------------------------------
    # Scenario-specific content-filter handling
    #
    # Scenario 745 contains wording that triggered Azure's
    # content-management filter.
    #
    # We protect only the specific sentence that caused the
    # filter to trigger. The sentence is restored verbatim
    # after conversion.
    #
    # No other scenario is affected.
    # --------------------------------------------------------

    if scenario_id == 745:

        import re

        protected_pattern = (
            r"The cat, feeling threatened, scratches John fiercely, "
            r"leaving him with deep wounds\."
        )

        match = re.search(
            protected_pattern,
            path1,
        )

        if match:

            protected_sentence = match.group(0)

            path1 = re.sub(
                protected_pattern,
                "[PROTECTED_SENTENCE]",
                path1,
                count=1,
            )

            print(
                "  Scenario 745: protected Azure-filter-triggering "
                "sentence before conversion."
            )

        else:

            raise ValueError(
                "Scenario 745 expected protected sentence was "
                "not found in SAE path 1."
            )

    # --------------------------------------------------------
    # Build user prompt
    # --------------------------------------------------------

    user_prompt = f"""
Convert the following SAE statements into AAE while preserving
their meaning exactly.

The input is already in statement form.

If [PROTECTED_SENTENCE] appears anywhere in the input, preserve
the placeholder exactly as written.

SAE STATEMENT:

{statement_text}

SAE PATH 1:

{path1}

SAE PATH 2:

{path2}

Return exactly this JSON structure:

{{
  "aae_statement_text": "...",
  "aae_statement_path1": "...",
  "aae_statement_path2": "..."
}}
"""

    # --------------------------------------------------------
    # Call model
    # --------------------------------------------------------

    response = client.responses.create(
        model=MODEL_DEPLOYMENT,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )

    text = response.output_text.strip()

    # --------------------------------------------------------
    # Handle accidental markdown fences
    # --------------------------------------------------------

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    result = json.loads(text)

    required = [
        "aae_statement_text",
        "aae_statement_path1",
        "aae_statement_path2",
    ]

    for key in required:

        if key not in result:
            raise ValueError(
                f"Missing field: {key}"
            )

        if not str(result[key]).strip():
            raise ValueError(
                f"Empty field: {key}"
            )

    # --------------------------------------------------------
    # Restore protected sentence verbatim
    # --------------------------------------------------------

    if protected_sentence is not None:

        converted_path1 = str(
            result["aae_statement_path1"]
        )

        placeholder_count = converted_path1.count(
            "[PROTECTED_SENTENCE]"
        )

        if placeholder_count != 1:
            raise ValueError(
                "Scenario 745 must contain exactly one "
                "[PROTECTED_SENTENCE] placeholder in "
                "AAE path 1."
            )

        result["aae_statement_path1"] = (
            converted_path1.replace(
                "[PROTECTED_SENTENCE]",
                protected_sentence,
                1,
            )
        )

        # Final integrity check.
        if protected_sentence not in result[
            "aae_statement_path1"
        ]:
            raise ValueError(
                "Protected sentence was not restored correctly."
            )

    return result


# ============================================================
# Connect to Microsoft Foundry
# ============================================================

print()
print("=" * 70)
print("CONNECTING TO MICROSOFT FOUNDRY")
print("=" * 70)

project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

foundry_openai = project.get_openai_client()

print(
    f"Connected to Foundry deployment: "
    f"{MODEL_DEPLOYMENT}"
)


# ============================================================
# Load existing canonical output
# ============================================================

print()
print("=" * 70)
print("LOADING EXISTING AAE CONVERSIONS")
print("=" * 70)

if os.path.exists(OUTPUT_FILE):

    existing = pd.read_csv(
        OUTPUT_FILE
    )

    print(
        f"Existing rows: {len(existing)}"
    )

    if "scenario_id" not in existing.columns:
        raise RuntimeError(
            "Existing output is missing scenario_id."
        )

    # --------------------------------------------------------
    # Clean duplicate histories
    # --------------------------------------------------------

    if existing["scenario_id"].duplicated().any():

        print(
            "Duplicate scenario histories detected."
        )

        existing["_statement_complete"] = (
            existing["aae_statement_text"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            &
            existing["aae_statement_path1"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            &
            existing["aae_statement_path2"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ).astype(int)

        existing["_completed_priority"] = (
            existing["conversion_status"]
            .astype(str)
            .str.lower()
            .eq("completed")
            .astype(int)
        )

        existing = (
            existing
            .sort_values(
                [
                    "scenario_id",
                    "_statement_complete",
                    "_completed_priority",
                ]
            )
            .drop_duplicates(
                subset=["scenario_id"],
                keep="last",
            )
            .drop(
                columns=[
                    "_statement_complete",
                    "_completed_priority",
                ]
            )
            .reset_index(drop=True)
        )

        print(
            f"After duplicate cleanup: "
            f"{len(existing)} rows"
        )

else:

    existing = pd.DataFrame()

    print(
        "No existing output file found."
    )


# ============================================================
# Determine VALID completed statement conversions
# ============================================================

if len(existing) > 0:

    statement_complete_mask = (
        existing["conversion_status"]
        .astype(str)
        .str.lower()
        .eq("completed")
        &
        existing["aae_statement_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        &
        existing["aae_statement_path1"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        &
        existing["aae_statement_path2"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    completed_ids = set(
        existing.loc[
            statement_complete_mask,
            "scenario_id"
        ].astype(int)
    )

else:

    completed_ids = set()


print()
print(
    f"Existing completed AAE STATEMENT conversions: "
    f"{len(completed_ids)}"
)


# ============================================================
# Diagnostics
# ============================================================

if len(existing) > 0:

    all_completed = set(
        existing.loc[
            existing["conversion_status"]
            .astype(str)
            .str.lower()
            .eq("completed"),
            "scenario_id"
        ].astype(int)
    )

    old_completed_but_not_statement = (
        all_completed - completed_ids
    )

    print(
        f"Completed rows that are NOT statement conversions: "
        f"{len(old_completed_but_not_statement)}"
    )


# ============================================================
# Determine pending scenarios
# ============================================================

pending = df[
    ~df["scenario_id"]
    .astype(int)
    .isin(completed_ids)
].copy()

print(
    f"Pending statement conversions: {len(pending)}"
)


# ============================================================
# Safety checks
# ============================================================

expected_pending = 250 - len(completed_ids)

if len(pending) != expected_pending:
    raise RuntimeError(
        "Pending count mismatch. "
        f"Expected {expected_pending}, "
        f"found {len(pending)}."
    )


if len(completed_ids) > 250:
    raise RuntimeError(
        "More completed statement conversions exist "
        "than total experimental scenarios."
    )


# ============================================================
# PILOT
# ============================================================

pilot = pending.head(PILOT_SIZE).copy()

print()
print("=" * 70)
print("PILOT CONFIGURATION")
print("=" * 70)

print(
    f"Processing {len(pilot)} scenarios."
)

print(
    f"Model deployment: {MODEL_DEPLOYMENT}"
)

print(
    f"PILOT_SIZE: {PILOT_SIZE}"
)

print(
    "Only pending AAE statement conversions "
    "will be processed."
)

print()

print(
    "Pilot scenario IDs:"
)

print(
    pilot["scenario_id"]
    .astype(int)
    .tolist()
)


# ============================================================
# Run pilot
# ============================================================

print()
print("=" * 70)
print(
    "RUNNING SAE STATEMENT → AAE STATEMENT CONVERSION PILOT"
)
print("=" * 70)


# Start from existing canonical dataset.
#
# Each processed scenario REPLACES the previous version.
# No duplicate scenario histories are created.

result_table = existing.copy()


for i, (_, row) in enumerate(
    pilot.iterrows(),
    start=1,
):

    scenario_id = int(
        row["scenario_id"]
    )

    print()
    print(
        f"[{i}/{len(pilot)}] "
        f"Scenario: {scenario_id}"
    )

    try:

        converted = convert_scenario(
            row,
            foundry_openai
        )

        new_record = {

            "scenario_id":
                scenario_id,

            "source":
                row["source"],

            "source_id":
                row["source_id"],

            "sae_statement_text":
                row["sae_statement_text"],

            "sae_statement_path1":
                row["sae_statement_path1"],

            "sae_statement_path2":
                row["sae_statement_path2"],

            "aae_statement_text":
                converted[
                    "aae_statement_text"
                ],

            "aae_statement_path1":
                converted[
                    "aae_statement_path1"
                ],

            "aae_statement_path2":
                converted[
                    "aae_statement_path2"
                ],

            "conversion_model":
                MODEL_DEPLOYMENT,

            "prompt_version":
                PROMPT_VERSION,

            "conversion_status":
                "completed",

            "validation_status":
                "pending",

            "validation_notes":
                "",
        }

        # ----------------------------------------------------
        # Remove previous version
        # ----------------------------------------------------

        result_table = result_table[
            result_table["scenario_id"]
            .astype(int)
            != scenario_id
        ]

        # ----------------------------------------------------
        # Add successful conversion
        # ----------------------------------------------------

        result_table = pd.concat(
            [
                result_table,
                pd.DataFrame([new_record]),
            ],
            ignore_index=True,
        )

        # ----------------------------------------------------
        # Keep canonical order
        # ----------------------------------------------------

        result_table = (
            result_table
            .sort_values("scenario_id")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        result_table.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print(
            "  Conversion completed."
        )

        print()
        print(
            "  AAE statement:"
        )

        print(
            converted[
                "aae_statement_text"
            ]
        )

        print()
        print(
            "  AAE path 1:"
        )

        print(
            converted[
                "aae_statement_path1"
            ]
        )

        print()
        print(
            "  AAE path 2:"
        )

        print(
            converted[
                "aae_statement_path2"
            ]
        )

    except Exception as e:

        print(
            f"  ERROR: {e}"
        )

        # ----------------------------------------------------
        # Preserve the scenario as an error row
        # ----------------------------------------------------

        error_record = {

            "scenario_id":
                scenario_id,

            "source":
                row["source"],

            "source_id":
                row["source_id"],

            "sae_statement_text":
                row["sae_statement_text"],

            "sae_statement_path1":
                row["sae_statement_path1"],

            "sae_statement_path2":
                row["sae_statement_path2"],

            "aae_statement_text":
                "",

            "aae_statement_path1":
                "",

            "aae_statement_path2":
                "",

            "conversion_model":
                MODEL_DEPLOYMENT,

            "prompt_version":
                PROMPT_VERSION,

            "conversion_status":
                "error",

            "validation_status":
                "pending",

            "validation_notes":
                str(e),
        }

        result_table = result_table[
            result_table["scenario_id"]
            .astype(int)
            != scenario_id
        ]

        result_table = pd.concat(
            [
                result_table,
                pd.DataFrame([error_record]),
            ],
            ignore_index=True,
        )

        result_table = (
            result_table
            .sort_values("scenario_id")
            .reset_index(drop=True)
        )

        result_table.to_csv(
            OUTPUT_FILE,
            index=False,
        )

    time.sleep(0.2)


# ============================================================
# Final validation
# ============================================================

output = pd.read_csv(
    OUTPUT_FILE
)

print()
print("=" * 70)
print("PILOT CONVERSION COMPLETE")
print("=" * 70)

print(
    f"Total rows: {len(output)}"
)

print(
    f"Unique scenarios: "
    f"{output['scenario_id'].nunique()}"
)

duplicate_count = (
    output["scenario_id"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate rows: {duplicate_count}"
)

print()

print(
    "Conversion status:"
)

print(
    output["conversion_status"]
    .value_counts()
)

# Count actual completed statement conversions

statement_complete_mask = (
    output["conversion_status"]
    .astype(str)
    .str.lower()
    .eq("completed")
    &
    output["aae_statement_text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
    &
    output["aae_statement_path1"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
    &
    output["aae_statement_path2"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

completed_statement_conversions = (
    statement_complete_mask.sum()
)

errors = (
    output["conversion_status"]
    .astype(str)
    .str.lower()
    .eq("error")
).sum()

print()
print(
    f"Completed AAE statement conversions: "
    f"{completed_statement_conversions}"
)

print(
    f"Errors: {errors}"
)

print()
print(
    f"Saved to:\n{OUTPUT_FILE}"
)


# ============================================================
# Next step
# ============================================================

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    f"Manually inspect the {len(pilot)} "
    "new pilot AAE statement conversions."
)

print()

print(
    "Then run:"
)

print(
    "python3 scripts/validate_aae_conversions.py"
)

print()

print(
    "DO NOT increase PILOT_SIZE until the pilot "
    "outputs have been manually validated."
)
