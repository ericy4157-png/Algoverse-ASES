import os
import re
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    "data/full/aae_conversion/aae_conversions_gpt5.csv"
)

VALIDATED_FILE = (
    "data/full/aae_conversion/aae_conversions_validated.csv"
)

REPORT_FILE = (
    "data/full/aae_conversion/validation_report.csv"
)

EXPECTED_SCENARIOS = 250


# ============================================================
# Helper functions
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_question_form(value):
    text = clean_text(value)
    return text.endswith("?")


def has_real_content(value):
    """
    Determines whether a field contains meaningful text
    rather than being empty or a structural placeholder.
    """
    text = clean_text(value)

    if not text:
        return False

    placeholders = {
        "n/a",
        "na",
        "none",
        "null",
        "...",
        "placeholder",
        "[placeholder]",
    }

    if text.lower() in placeholders:
        return False

    return True


def normalize_for_comparison(text):
    """
    Conservative normalization used only for detecting
    exact SAE/AAE copying.
    """
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_meaningful_dialect_change(sae_text, aae_text):
    """
    Determines whether the AAE output differs from SAE.

    This is diagnostic only.

    An unchanged field is NOT considered a validation error,
    because legitimate AAE conversion can leave portions
    of the text unchanged.
    """
    return (
        normalize_for_comparison(sae_text)
        != normalize_for_comparison(aae_text)
    )


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("LOADING AAE STATEMENT CONVERSIONS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows.")

if len(df) != EXPECTED_SCENARIOS:
    raise RuntimeError(
        f"Expected {EXPECTED_SCENARIOS} rows, "
        f"but found {len(df)}."
    )


# ============================================================
# Validate schema
# ============================================================

required_columns = [
    "scenario_id",
    "source",
    "source_id",
    "sae_statement_text",
    "sae_statement_path1",
    "sae_statement_path2",
    "aae_statement_text",
    "aae_statement_path1",
    "aae_statement_path2",
    "conversion_status",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# Identify completed statement conversions
# ============================================================

print()
print("=" * 70)
print("IDENTIFYING AAE STATEMENT CONVERSIONS")
print("=" * 70)

completed_mask = (
    df["conversion_status"]
    .astype(str)
    .str.lower()
    .eq("completed")
)

statement_fields = [
    "aae_statement_text",
    "aae_statement_path1",
    "aae_statement_path2",
]

statement_content_mask = (
    df[statement_fields]
    .fillna("")
    .astype(str)
    .apply(
        lambda col: col.str.strip().ne("")
    )
    .all(axis=1)
)

statement_conversions = df[
    completed_mask & statement_content_mask
].copy()

legacy_or_incomplete = df[
    ~(completed_mask & statement_content_mask)
].copy()

print(
    f"Completed AAE statement conversions: "
    f"{len(statement_conversions)}"
)

print(
    f"Remaining legacy/incomplete rows: "
    f"{len(legacy_or_incomplete)}"
)

print(
    f"Total scenarios: "
    f"{len(df)}"
)


# ============================================================
# Ensure conversions exist
# ============================================================

if len(statement_conversions) == 0:
    raise RuntimeError(
        "No completed AAE statement conversions "
        "were found to validate."
    )


# ============================================================
# Validation
# ============================================================

print()
print("=" * 70)
print("VALIDATING COMPLETED AAE STATEMENT CONVERSIONS")
print("=" * 70)

validation_records = []


for _, row in statement_conversions.iterrows():

    scenario_id = int(row["scenario_id"])

    errors = []
    warnings = []
    diagnostics = []


    # --------------------------------------------------------
    # SAE inputs
    # --------------------------------------------------------

    sae_text = clean_text(
        row["sae_statement_text"]
    )

    sae_path1 = clean_text(
        row["sae_statement_path1"]
    )

    sae_path2 = clean_text(
        row["sae_statement_path2"]
    )


    # --------------------------------------------------------
    # AAE statement
    # --------------------------------------------------------

    aae_text = clean_text(
        row["aae_statement_text"]
    )

    if not has_real_content(aae_text):
        errors.append(
            "missing_aae_statement_text"
        )

    if is_question_form(aae_text):
        errors.append(
            "aae_statement_text_question_form"
        )


    # --------------------------------------------------------
    # AAE path 1
    # --------------------------------------------------------

    aae_path1 = clean_text(
        row["aae_statement_path1"]
    )

    if not has_real_content(aae_path1):
        errors.append(
            "missing_aae_statement_path1"
        )

    if is_question_form(aae_path1):
        errors.append(
            "aae_statement_path1_question_form"
        )


    # --------------------------------------------------------
    # AAE path 2
    # --------------------------------------------------------

    aae_path2 = clean_text(
        row["aae_statement_path2"]
    )

    if not has_real_content(aae_path2):
        errors.append(
            "missing_aae_statement_path2"
        )

    if is_question_form(aae_path2):
        errors.append(
            "aae_statement_path2_question_form"
        )


    # ========================================================
    # Copying diagnostics
    # ========================================================
    #
    # IMPORTANT:
    #
    # Identical SAE/AAE text is NOT an error.
    #
    # It is also NOT a warning anymore.
    #
    # We simply record it as a diagnostic so we can measure
    # how frequently the model leaves text unchanged.
    #
    # ========================================================

    if not contains_meaningful_dialect_change(
        sae_text,
        aae_text
    ):
        diagnostics.append(
            "aae_statement_text_identical_to_sae"
        )

    if not contains_meaningful_dialect_change(
        sae_path1,
        aae_path1
    ):
        diagnostics.append(
            "aae_statement_path1_identical_to_sae"
        )

    if not contains_meaningful_dialect_change(
        sae_path2,
        aae_path2
    ):
        diagnostics.append(
            "aae_statement_path2_identical_to_sae"
        )


    # ========================================================
    # Suspicious AI/meta language
    # ========================================================

    suspicious_phrases = [
        "as an ai",
        "as an artificial intelligence",
        "i can't",
        "i cannot",
        "here's the",
        "here is the",
        "certainly,",
        "of course,",
    ]

    combined_aae = (
        f"{aae_text} "
        f"{aae_path1} "
        f"{aae_path2}"
    ).lower()

    for phrase in suspicious_phrases:

        if phrase in combined_aae:

            warnings.append(
                f"suspicious_phrase:{phrase}"
            )


    # ========================================================
    # Question-mark diagnostics
    # ========================================================

    for field_name, value in [
        ("aae_statement_text", aae_text),
        ("aae_statement_path1", aae_path1),
        ("aae_statement_path2", aae_path2),
    ]:

        if "?" in value:

            warnings.append(
                f"question_mark_present:{field_name}"
            )


    # ========================================================
    # Determine validation status
    # ========================================================

    if errors:
        status = "flagged"

    elif warnings:
        status = "warning"

    else:
        status = "pass"


    # ========================================================
    # Save validation record
    # ========================================================

    validation_records.append(
        {
            "scenario_id": scenario_id,
            "status": status,
            "errors": "; ".join(errors),
            "warnings": "; ".join(warnings),
            "diagnostics": "; ".join(diagnostics),
        }
    )


# ============================================================
# Validation DataFrame
# ============================================================

validation_df = pd.DataFrame(
    validation_records
)


# ============================================================
# Print validation results
# ============================================================

print()
print("=" * 70)
print(
    "VALIDATION RESULTS — COMPLETED STATEMENT "
    "CONVERSIONS ONLY"
)
print("=" * 70)

status_counts = (
    validation_df["status"]
    .value_counts()
)

print(status_counts)

print()

print(
    f"Validated statement conversions: "
    f"{len(validation_df)}"
)

print(
    f"Passed: "
    f"{(validation_df['status'] == 'pass').sum()}"
)

print(
    f"Warnings: "
    f"{(validation_df['status'] == 'warning').sum()}"
)

print(
    f"Flagged: "
    f"{(validation_df['status'] == 'flagged').sum()}"
)


# ============================================================
# Error counts
# ============================================================

print()
print("=" * 70)
print("ERROR COUNTS")
print("=" * 70)

error_counter = {}

for errors in validation_df["errors"]:

    if not errors:
        continue

    for error in errors.split("; "):

        if not error:
            continue

        error_counter[error] = (
            error_counter.get(error, 0) + 1
        )


if error_counter:

    for error, count in sorted(
        error_counter.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"{error:<55} {count}"
        )

else:

    print("No validation errors.")


# ============================================================
# Warning counts
# ============================================================

print()
print("=" * 70)
print("WARNING COUNTS")
print("=" * 70)

warning_counter = {}

for warnings in validation_df["warnings"]:

    if not warnings:
        continue

    for warning in warnings.split("; "):

        if not warning:
            continue

        warning_counter[warning] = (
            warning_counter.get(warning, 0) + 1
        )


if warning_counter:

    for warning, count in sorted(
        warning_counter.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"{warning:<55} {count}"
        )

else:

    print("No validation warnings.")


# ============================================================
# Copying diagnostics
# ============================================================

print()
print("=" * 70)
print("UNCHANGED SAE/AAE FIELD DIAGNOSTICS")
print("=" * 70)

diagnostic_counter = {}

for diagnostics in validation_df["diagnostics"]:

    if not diagnostics:
        continue

    for diagnostic in diagnostics.split("; "):

        if not diagnostic:
            continue

        diagnostic_counter[diagnostic] = (
            diagnostic_counter.get(diagnostic, 0) + 1
        )


if diagnostic_counter:

    for diagnostic, count in sorted(
        diagnostic_counter.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"{diagnostic:<55} {count}"
        )

else:

    print("No unchanged SAE/AAE fields detected.")


# ============================================================
# Dataset coverage
# ============================================================

print()
print("=" * 70)
print("DATASET COVERAGE")
print("=" * 70)

print(
    f"Total experimental scenarios: "
    f"{len(df)}"
)

print(
    f"Completed AAE statement conversions: "
    f"{len(statement_conversions)}"
)

print(
    f"Remaining scenarios requiring conversion: "
    f"{len(legacy_or_incomplete)}"
)


# ============================================================
# Conversion coverage percentage
# ============================================================

coverage = (
    len(statement_conversions)
    / EXPECTED_SCENARIOS
    * 100
)

print(
    f"Conversion coverage: "
    f"{coverage:.1f}%"
)


# ============================================================
# Save validation report
# ============================================================

validation_df.to_csv(
    REPORT_FILE,
    index=False,
)

print()
print(
    f"Validation report saved to:\n"
    f"{REPORT_FILE}"
)


# ============================================================
# Save validated dataset
# ============================================================
#
# IMPORTANT:
#
# We preserve ALL 250 rows.
#
# validation_status is only assigned to rows that have
# actually undergone AAE statement conversion.
#
# ============================================================

output_df = df.copy()

output_df["validation_status"] = "not_validated"
output_df["validation_notes"] = ""


for _, validation_row in validation_df.iterrows():

    scenario_id = int(
        validation_row["scenario_id"]
    )

    mask = (
        output_df["scenario_id"]
        .astype(int)
        .eq(scenario_id)
    )

    output_df.loc[
        mask,
        "validation_status"
    ] = validation_row["status"]

    output_df.loc[
        mask,
        "validation_notes"
    ] = (
        f"Errors: "
        f"{validation_row['errors']} | "
        f"Warnings: "
        f"{validation_row['warnings']} | "
        f"Diagnostics: "
        f"{validation_row['diagnostics']}"
    )


output_df.to_csv(
    VALIDATED_FILE,
    index=False,
)

print(
    f"Validated dataset saved to:\n"
    f"{VALIDATED_FILE}"
)


# ============================================================
# Final decision
# ============================================================

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

flagged_count = (
    validation_df["status"]
    .eq("flagged")
    .sum()
)

warning_count = (
    validation_df["status"]
    .eq("warning")
    .sum()
)


if flagged_count > 0:

    print(
        f"FLAGGED STATEMENT CONVERSIONS REMAIN: "
        f"{flagged_count}"
    )

    print()

    print(
        "Inspect the flagged scenarios before scaling."
    )

elif warning_count > 0:

    print(
        "No hard validation failures."
    )

    print(
        f"{warning_count} statement conversions "
        f"have warnings requiring manual review."
    )

    print()

    print(
        "Review the warnings before scaling."
    )

else:

    print(
        "ALL COMPLETED STATEMENT CONVERSIONS "
        "PASSED HARD VALIDATION."
    )

    print()

    print(
        f"{len(statement_conversions)} / "
        f"{EXPECTED_SCENARIOS} scenarios converted."
    )

    print(
        f"{len(legacy_or_incomplete)} scenarios "
        f"remain to be converted."
    )

    print()

    print(
        "Pilot validation is clean."
    )


print()

print(
    "Do NOT run model evaluations until the full "
    "experimental dataset has been converted and validated."
)