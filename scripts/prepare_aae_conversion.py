import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

STATEMENT_QUEUE_FILE = (
    "data/full/aae_conversion/statement_conversion_queue.csv"
)

OUTPUT_DIR = "data/full/aae_conversion"

AAE_QUEUE_FILE = (
    f"{OUTPUT_DIR}/aae_statement_conversion_queue.csv"
)

PILOT_FILE = (
    f"{OUTPUT_DIR}/aae_pilot_conversion_anchors.csv"
)

EXPECTED_SAMPLE_SIZE = 250

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load completed SAE statement conversion queue
# ============================================================

print("=" * 70)
print("LOADING COMPLETED SAE STATEMENT CONVERSION QUEUE")
print("=" * 70)

if not os.path.exists(STATEMENT_QUEUE_FILE):
    raise RuntimeError(
        f"Statement conversion queue not found: "
        f"{STATEMENT_QUEUE_FILE}"
    )

df = pd.read_csv(STATEMENT_QUEUE_FILE)

print(f"Loaded {len(df)} scenarios.")


# ============================================================
# Validate required columns
# ============================================================

print()
print("=" * 70)
print("VALIDATING STATEMENT QUEUE SCHEMA")
print("=" * 70)

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
        "Statement queue is missing required columns: "
        f"{missing}"
    )

print("Statement queue schema validated.")


# ============================================================
# Validate total sample
# ============================================================

print()
print("=" * 70)
print("VALIDATING EXPERIMENTAL SAMPLE")
print("=" * 70)

print(
    f"Expected scenarios: {EXPECTED_SAMPLE_SIZE}"
)

print(
    f"Actual scenarios: {len(df)}"
)

if len(df) != EXPECTED_SAMPLE_SIZE:
    raise RuntimeError(
        f"Expected exactly {EXPECTED_SAMPLE_SIZE} scenarios, "
        f"but found {len(df)}."
    )


# ============================================================
# Validate scenario IDs
# ============================================================

duplicate_ids = (
    df["scenario_id"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate scenario IDs: {duplicate_ids}"
)

if duplicate_ids > 0:
    raise RuntimeError(
        "Duplicate scenario IDs detected."
    )


# ============================================================
# Check statement conversion status
# ============================================================

print()
print("=" * 70)
print("CHECKING SAE STATEMENT CONVERSION STATUS")
print("=" * 70)

status_counts = (
    df["statement_conversion_status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .value_counts()
)

print(status_counts)


completed_mask = (
    df["statement_conversion_status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("completed")
)

completed_count = completed_mask.sum()

print()
print(
    f"Completed statement conversions: "
    f"{completed_count}"
)

if completed_count != EXPECTED_SAMPLE_SIZE:
    print()
    print(
        "WARNING: Not every scenario has a completed "
        "SAE statement conversion."
    )


# ============================================================
# Identify failed/incomplete scenarios
# ============================================================

incomplete = df[~completed_mask].copy()

if len(incomplete) > 0:

    print()
    print("=" * 70)
    print("INCOMPLETE STATEMENT CONVERSIONS")
    print("=" * 70)

    for _, row in incomplete.iterrows():

        print(
            f"Scenario {row['scenario_id']}: "
            f"{row['statement_conversion_status']}"
        )

        print(
            f"Notes: "
            f"{row['statement_validation_notes']}"
        )

    print()
    print(
        "These scenarios will NOT be included in the "
        "AAE conversion queue."
    )


# ============================================================
# Validate completed statement fields
# ============================================================

print()
print("=" * 70)
print("VALIDATING SAE STATEMENT OUTPUTS")
print("=" * 70)

completed = df[completed_mask].copy()

statement_columns = [
    "sae_statement_text",
    "sae_statement_path1",
    "sae_statement_path2",
]

for column in statement_columns:

    empty_count = (
        completed[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"{column}: {empty_count} empty"
    )

    if empty_count > 0:
        raise RuntimeError(
            f"Completed statement queue contains empty "
            f"values in {column}."
        )


# ============================================================
# Validate that statements are not questions
# ============================================================

print()
print("=" * 70)
print("CHECKING FOR QUESTION-FORM OUTPUTS")
print("=" * 70)

for column in statement_columns:

    question_count = (
        completed[column]
        .fillna("")
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


# ============================================================
# Build AAE statement conversion queue
# ============================================================

print()
print("=" * 70)
print("BUILDING AAE STATEMENT CONVERSION QUEUE")
print("=" * 70)

aae_queue = pd.DataFrame({

    # --------------------------------------------------------
    # Scenario identifiers
    # --------------------------------------------------------

    "scenario_id":
        completed["scenario_id"].values,

    "source":
        completed["source"].values,

    "source_id":
        completed["source_id"].values,

    # --------------------------------------------------------
    # Original SAE material
    # --------------------------------------------------------

    "original_sae_text":
        completed["original_sae_text"].values,

    "original_sae_path1":
        completed["original_sae_path1"].values,

    "original_sae_path2":
        completed["original_sae_path2"].values,

    # --------------------------------------------------------
    # SAE statement material
    # --------------------------------------------------------

    "sae_statement_text":
        completed["sae_statement_text"].values,

    "sae_statement_path1":
        completed["sae_statement_path1"].values,

    "sae_statement_path2":
        completed["sae_statement_path2"].values,

    # --------------------------------------------------------
    # AAE conversion outputs
    # --------------------------------------------------------

    "aae_statement_text": "",

    "aae_statement_path1": "",

    "aae_statement_path2": "",

    # --------------------------------------------------------
    # AAE conversion status
    # --------------------------------------------------------

    "aae_conversion_status":
        "pending",

    "aae_validation_status":
        "pending",

    "aae_validation_notes":
        "",
})


# ============================================================
# Final AAE queue validation
# ============================================================

print()
print("=" * 70)
print("FINAL AAE QUEUE VALIDATION")
print("=" * 70)

print(
    f"Rows: {len(aae_queue)}"
)

print(
    f"Expected rows: {EXPECTED_SAMPLE_SIZE}"
)

duplicate_aae_ids = (
    aae_queue["scenario_id"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate scenario IDs: {duplicate_aae_ids}"
)

if len(aae_queue) != EXPECTED_SAMPLE_SIZE:
    raise RuntimeError(
        "AAE queue does not contain exactly 250 scenarios."
    )

if duplicate_aae_ids > 0:
    raise RuntimeError(
        "AAE queue contains duplicate scenario IDs."
    )


# ------------------------------------------------------------
# Validate SAE statement fields in final queue
# ------------------------------------------------------------

for column in statement_columns:

    empty_count = (
        aae_queue[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"Empty {column}: {empty_count}"
    )

    if empty_count > 0:
        raise RuntimeError(
            f"AAE queue contains empty {column} values."
        )


# ============================================================
# Save AAE queue
# ============================================================

aae_queue.to_csv(
    AAE_QUEUE_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("AAE STATEMENT CONVERSION QUEUE CREATED")
print("=" * 70)

print(
    f"Total AAE conversion scenarios: "
    f"{len(aae_queue)}"
)

print()

print("By source:")

print(
    aae_queue["source"].value_counts()
)

print()

print(
    "Pending AAE conversions: "
    f"{(aae_queue['aae_conversion_status'] == 'pending').sum()}"
)

print()

print(
    "Saved to:"
)

print(
    AAE_QUEUE_FILE
)


# ============================================================
# Pilot anchors
# ============================================================

print()
print("=" * 70)
print("AAE PILOT ANCHORS")
print("=" * 70)

if os.path.exists(PILOT_FILE):

    pilot = pd.read_csv(PILOT_FILE)

    print(
        f"Found existing AAE pilot anchors: "
        f"{len(pilot)}"
    )

    print(
        "Existing pilot anchors were NOT modified."
    )

else:

    print(
        "No AAE pilot anchor file found."
    )


# ============================================================
# Final message
# ============================================================

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "The original 250-scenario experimental sample "
    "has been preserved."
)

print(
    "Only scenarios with completed SAE statement "
    "conversions were included."
)

print(
    "The AAE conversion queue uses the SAE STATEMENT "
    "fields, not the original question-form stimuli."
)

print(
    "Original SAE stimuli remain preserved."
)

print()
print(
    "Next: run the SAE STATEMENT → AAE STATEMENT "
    "conversion pilot."
)

print(
    "Do NOT run the full AAE conversion yet."
)