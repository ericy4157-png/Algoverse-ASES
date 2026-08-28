import os
import time
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient


# ============================================================
# Configuration
# ============================================================

QUEUE_FILE = (
    "data/full/aae_conversion/statement_conversion_queue.csv"
)

OUTPUT_FILE = (
    "data/full/aae_conversion/statement_conversion_queue.csv"
)

MODEL_DEPLOYMENT = "gpt-5.4-1"

# IMPORTANT:
# Start with 5 for the pilot.
# Change to 250 ONLY after manually validating the pilot.
PILOT_SIZE = 1

# Small delay between requests.
REQUEST_DELAY_SECONDS = 1


# ============================================================
# Foundry project configuration
# ============================================================

PROJECT_ENDPOINT = (
    "https://ericy4158-7533-resource.services.ai.azure.com/"
    "api/projects/ericy4158-7533"
)


# ============================================================
# Conversion prompt
# ============================================================

SYSTEM_PROMPT = """
You are performing a controlled linguistic transformation for a
research dataset.

Your task is to convert SAE (Standard American English) scenario
stimuli into natural, grammatical declarative statements suitable
for later dialect conversion.

The goal is to remove the QUESTION FORMAT while preserving the
underlying scenario and dilemma as faithfully as possible.

CRITICAL RULES:

1. Preserve the underlying meaning exactly.

2. Do NOT answer the question.

3. Do NOT imply which option is morally correct.

4. Do NOT change the facts of the scenario.

5. Do NOT change the characters, motivations, circumstances,
   alternatives, or consequences.

6. Do NOT add new information.

7. Do NOT remove substantive information.

8. Do NOT introduce stereotypes or racialized content.

9. Do NOT alter the substantive situation.

10. Do NOT make the language unnecessarily more formal.

11. Preserve the original first-person, second-person, or
    third-person perspective.

12. The output must be a declarative description of the dilemma,
    not a question.

13. If the original asks a question such as:
       "Should Sarah continue dating John?"
    rewrite it as a decision statement such as:
       "Sarah must decide whether to continue dating John."

14. If the original asks an ethical question such as:
       "Is it ethical to ask them to renew their vows?"
    DO NOT convert it into:
       "It is ethical to ask them to renew their vows."
    Instead, preserve the unresolved ethical decision, for example:
       "The situation requires a decision about whether it is
       appropriate to ask them to renew their vows."

15. Do NOT resolve an ethical dilemma merely because the original
    was phrased as a question.

16. For Path 1 and Path 2, preserve the original alternative and
    consequence. Convert them into declarative statements only.
    Do not change which option is represented by either path.

17. Keep the wording as close to the original as naturally possible.
    This is a controlled transformation, not a creative rewrite.

Return exactly three lines:

STATEMENT_TEXT: <converted scenario statement>
STATEMENT_PATH1: <converted path 1>
STATEMENT_PATH2: <converted path 2>

Do not include any additional commentary.
"""


# ============================================================
# Helpers
# ============================================================

def extract_field(text, field_name):
    """
    Extract a field from the model's three-line response.
    """
    prefix = f"{field_name}:"

    for line in text.splitlines():
        line = line.strip()

        if line.startswith(prefix):
            return line[len(prefix):].strip()

    return ""


def validate_conversion(
    original_text,
    original_path1,
    original_path2,
    statement_text,
    statement_path1,
    statement_path2,
):
    """
    Basic automated checks.

    These checks do NOT replace human validation.
    """

    errors = []

    if not statement_text:
        errors.append("Empty statement_text")

    if not statement_path1:
        errors.append("Empty statement_path1")

    if not statement_path2:
        errors.append("Empty statement_path2")

    if statement_text.endswith("?"):
        errors.append(
            "Statement text still appears to be a question"
        )

    if statement_path1.endswith("?"):
        errors.append(
            "Path1 still appears to be a question"
        )

    if statement_path2.endswith("?"):
        errors.append(
            "Path2 still appears to be a question"
        )

    # Make sure the scenario was actually transformed.
    if statement_text.strip() == original_text.strip():
        errors.append(
            "Statement text is identical to original SAE text"
        )

    return errors


# ============================================================
# Load queue
# ============================================================

print("=" * 70)
print("LOADING SAE STATEMENT CONVERSION QUEUE")
print("=" * 70)

if not os.path.exists(QUEUE_FILE):
    raise RuntimeError(
        f"Queue file not found: {QUEUE_FILE}"
    )

df = pd.read_csv(QUEUE_FILE)

# Force output columns to string dtype.
# Empty CSV columns may otherwise be inferred as float64,
# which prevents us from writing generated text into them.

text_output_columns = [
    "sae_statement_text",
    "sae_statement_path1",
    "sae_statement_path2",
    "statement_validation_notes",
]

for column in text_output_columns:
    df[column] = (
        df[column]
        .fillna("")
        .astype("string")
    )

print(f"Loaded {len(df)} scenarios.")


# ============================================================
# Validate required columns
# ============================================================

required_columns = [
    "scenario_id",
    "source",
    "source_id",
    "original_sae_text",
    "original_sae_path1",
    "original_sae_path2",
    "sae_statement_text",
    "sae_statement_path1",
    "sae_statement_path2",
    "statement_conversion_status",
    "statement_validation_status",
    "statement_validation_notes",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        f"Queue is missing required columns: {missing}"
    )


# ============================================================
# Select pending rows
# ============================================================

pending_mask = (
    df["statement_conversion_status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("pending")
)

pending_indices = df.index[pending_mask].tolist()

print(
    f"Pending conversions: {len(pending_indices)}"
)

if not pending_indices:
    print("No pending conversions found.")
    raise SystemExit(0)


# ============================================================
# Pilot limit
# ============================================================

indices_to_process = pending_indices[:PILOT_SIZE]

print()
print("=" * 70)
print("PILOT CONFIGURATION")
print("=" * 70)

print(
    f"Processing {len(indices_to_process)} scenarios."
)

print(
    f"Model deployment: {MODEL_DEPLOYMENT}"
)

print(
    "This is a PILOT. The remaining scenarios will not be processed."
)


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

openai = project.get_openai_client()

print(
    f"Connected to Foundry deployment: {MODEL_DEPLOYMENT}"
)


# ============================================================
# Process pilot
# ============================================================

print()
print("=" * 70)
print("RUNNING SAE → STATEMENT CONVERSION PILOT")
print("=" * 70)


for counter, index in enumerate(
    indices_to_process,
    start=1,
):

    row = df.loc[index]

    scenario_id = row["scenario_id"]

    print()
    print(
        f"[{counter}/{len(indices_to_process)}] "
        f"Scenario: {scenario_id}"
    )

    user_prompt = f"""
Convert the following SAE scenario into a natural declarative
statement according to all rules in the system instructions.

ORIGINAL SAE SCENARIO:

{row["original_sae_text"]}

ORIGINAL SAE PATH 1:

{row["original_sae_path1"]}

ORIGINAL SAE PATH 2:

{row["original_sae_path2"]}
"""

    try:

        response = openai.responses.create(
            model=MODEL_DEPLOYMENT,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )

        output = response.output_text.strip()

        statement_text = extract_field(
            output,
            "STATEMENT_TEXT",
        )

        statement_path1 = extract_field(
            output,
            "STATEMENT_PATH1",
        )

        statement_path2 = extract_field(
            output,
            "STATEMENT_PATH2",
        )

        errors = validate_conversion(
            row["original_sae_text"],
            row["original_sae_path1"],
            row["original_sae_path2"],
            statement_text,
            statement_path1,
            statement_path2,
        )

        if errors:

            df.at[
                index,
                "statement_conversion_status"
            ] = "failed_validation"

            df.at[
                index,
                "statement_validation_status"
            ] = "failed"

            df.at[
                index,
                "statement_validation_notes"
            ] = "; ".join(errors)

            print(
                "  Conversion produced validation errors:"
            )

            for error in errors:
                print(
                    f"    - {error}"
                )

        else:

            df.at[
                index,
                "sae_statement_text"
            ] = statement_text

            df.at[
                index,
                "sae_statement_path1"
            ] = statement_path1

            df.at[
                index,
                "sae_statement_path2"
            ] = statement_path2

            df.at[
                index,
                "statement_conversion_status"
            ] = "completed"

            df.at[
                index,
                "statement_validation_status"
            ] = "pending_human_review"

            df.at[
                index,
                "statement_validation_notes"
            ] = ""

            print("  Conversion completed.")

            print(
                f"  Statement: {statement_text}"
            )

            print(
                f"  Path 1: {statement_path1}"
            )

            print(
                f"  Path 2: {statement_path2}"
            )

    except Exception as e:

        df.at[
            index,
            "statement_conversion_status"
        ] = "error"

        df.at[
            index,
            "statement_validation_status"
        ] = "error"

        df.at[
            index,
            "statement_validation_notes"
        ] = str(e)

        print(
            f"  ERROR: {type(e).__name__}: {e}"
        )

    # Save after every scenario so progress isn't lost.

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    time.sleep(
        REQUEST_DELAY_SECONDS
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("PILOT CONVERSION COMPLETE")
print("=" * 70)

completed = (
    df["statement_conversion_status"]
    .eq("completed")
    .sum()
)

failed = (
    df["statement_conversion_status"]
    .isin(
        [
            "failed_validation",
            "error",
        ]
    )
    .sum()
)

pending = (
    df["statement_conversion_status"]
    .eq("pending")
    .sum()
)

print(
    f"Completed conversions: {completed}"
)

print(
    f"Failed/error conversions: {failed}"
)

print(
    f"Still pending: {pending}"
)

print()

print(
    f"Saved to: {OUTPUT_FILE}"
)

print()

print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "Manually inspect the 5 pilot conversions."
)

print(
    "Do NOT change PILOT_SIZE to 250 until the pilot "
    "outputs have been validated."
)