import os
import re
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    "data/full/aae_conversion/aae_conversions_gpt5.csv"
)

OUTPUT_DIR = (
    "data/full/aae_conversion"
)

VALIDATED_FILE = (
    f"{OUTPUT_DIR}/aae_conversions_validated.csv"
)

REPORT_FILE = (
    f"{OUTPUT_DIR}/validation_report.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# Load
# ============================================================

print("=" * 70)
print("LOADING AAE CONVERSIONS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows.")

# ============================================================
# Expected canonical columns
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
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns: "
        + str(missing_columns)
    )

# ============================================================
# Helpers
# ============================================================

def normalize(text):
    """
    Normalize text for exact/near-exact comparison.
    """
    return re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ).strip()


def word_count(text):
    """
    Count whitespace-separated words.
    """
    return len(str(text).split())


def extract_numbers(text):
    """
    Extract numeric values while ignoring structural
    PATH labels.
    """
    text = str(text)

    text = re.sub(
        r"\bPATH\s+[12]\s*:",
        "",
        text,
        flags=re.IGNORECASE
    )

    return re.findall(
        r"\b\d+(?:\.\d+)?\b",
        text
    )


def is_placeholder_path(text):
    """
    Detect path fields that contain only a structural
    placeholder rather than actual prose.
    """

    normalized = normalize(text)

    placeholder_patterns = [
        r"^option[_\s-]?a$",
        r"^option[_\s-]?b$",
        r"^path\s*1$",
        r"^path\s*2$",
        r"^path\s*1\s*:$",
        r"^path\s*2\s*:$",
    ]

    return any(
        re.fullmatch(pattern, normalized)
        for pattern in placeholder_patterns
    )


def has_real_content(text):
    """
    True when the field contains actual text.
    """

    if pd.isna(text):
        return False

    value = str(text).strip()

    if not value:
        return False

    if value.lower() == "nan":
        return False

    return True


def has_real_path_content(text):
    """
    True when a path contains actual scenario/consequence
    prose rather than only a structural placeholder.
    """

    if not has_real_content(text):
        return False

    return not is_placeholder_path(text)


# ============================================================
# Basic dataset checks
# ============================================================

print()
print("=" * 70)
print("DATASET STRUCTURE CHECK")
print("=" * 70)

if len(df) != 250:
    raise RuntimeError(
        f"Expected exactly 250 scenarios, found {len(df)}."
    )

unique_ids = df["scenario_id"].nunique()

if unique_ids != 250:
    raise RuntimeError(
        f"Expected 250 unique scenarios, found {unique_ids}."
    )

duplicate_count = df["scenario_id"].duplicated().sum()

if duplicate_count != 0:
    raise RuntimeError(
        f"Found {duplicate_count} duplicate scenario rows."
    )

print("Rows: 250")
print("Unique scenarios: 250")
print("Duplicate rows: 0")

# ============================================================
# Validate each scenario
# ============================================================

reports = []

for _, row in df.iterrows():

    scenario_id = int(row["scenario_id"])

    errors = []
    warnings = []

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    for field in [
        "aae_statement_text",
        "aae_statement_path1",
        "aae_statement_path2",
    ]:

        if not has_real_content(row[field]):

            errors.append(
                f"missing_{field}"
            )

    # --------------------------------------------------------
    # Conversion status
    # --------------------------------------------------------

    if (
        str(row["conversion_status"])
        .strip()
        .lower()
        != "completed"
    ):

        errors.append(
            "conversion_not_completed"
        )

    # --------------------------------------------------------
    # Structural placeholder paths
    # --------------------------------------------------------

    for path in ["1", "2"]:

        sae_field = f"sae_statement_path{path}"
        aae_field = f"aae_statement_path{path}"

        sae = row[sae_field]
        aae = row[aae_field]

        sae_placeholder = is_placeholder_path(sae)
        aae_placeholder = is_placeholder_path(aae)

        if sae_placeholder and aae_placeholder:

            warnings.append(
                f"path{path}_structural_placeholder"
            )

        elif not has_real_path_content(aae):

            errors.append(
                f"aae_statement_path{path}_missing_real_content"
            )

    # --------------------------------------------------------
    # Identical statement text
    # --------------------------------------------------------

    if (
        has_real_content(row["sae_statement_text"])
        and has_real_content(row["aae_statement_text"])
        and normalize(row["sae_statement_text"])
        == normalize(row["aae_statement_text"])
    ):

        warnings.append(
            "aae_statement_text_identical_to_sae"
        )

    # --------------------------------------------------------
    # Identical path content
    # --------------------------------------------------------

    for path in ["1", "2"]:

        sae = row[f"sae_statement_path{path}"]
        aae = row[f"aae_statement_path{path}"]

        sae_real = has_real_path_content(sae)
        aae_real = has_real_path_content(aae)

        if (
            sae_real
            and aae_real
            and normalize(sae)
            == normalize(aae)
        ):

            warnings.append(
                f"aae_statement_path{path}_real_content_identical_to_sae"
            )

    # --------------------------------------------------------
    # Statement length
    # --------------------------------------------------------

    sae_words = word_count(
        row["sae_statement_text"]
    )

    aae_words = word_count(
        row["aae_statement_text"]
    )

    if sae_words > 0:

        ratio = aae_words / sae_words

        if ratio < 0.50:

            warnings.append(
                "aae_statement_text_much_shorter"
            )

        elif ratio > 1.50:

            warnings.append(
                "aae_statement_text_much_longer"
            )

    # --------------------------------------------------------
    # Path length
    # --------------------------------------------------------

    for path in ["1", "2"]:

        sae = row[f"sae_statement_path{path}"]
        aae = row[f"aae_statement_path{path}"]

        if (
            has_real_path_content(sae)
            and has_real_path_content(aae)
        ):

            sae_len = word_count(sae)
            aae_len = word_count(aae)

            if sae_len > 0:

                ratio = aae_len / sae_len

                if ratio < 0.50:

                    warnings.append(
                        f"path{path}_much_shorter"
                    )

                elif ratio > 1.50:

                    warnings.append(
                        f"path{path}_much_longer"
                    )

    # --------------------------------------------------------
    # Number preservation
    # --------------------------------------------------------

    field_pairs = [
        (
            "sae_statement_text",
            "aae_statement_text"
        ),
        (
            "sae_statement_path1",
            "aae_statement_path1"
        ),
        (
            "sae_statement_path2",
            "aae_statement_path2"
        ),
    ]

    for sae_field, aae_field in field_pairs:

        sae_numbers = sorted(
            extract_numbers(row[sae_field])
        )

        aae_numbers = sorted(
            extract_numbers(row[aae_field])
        )

        if sae_numbers != aae_numbers:

            errors.append(
                f"numbers_changed_{sae_field}"
            )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if errors:

        status = "flagged"

    elif warnings:

        status = "warning"

    else:

        status = "pass"

    reports.append({

        "scenario_id": scenario_id,

        "source": row["source"],

        "status": status,

        "error_count": len(errors),

        "warning_count": len(warnings),

        "errors": "; ".join(errors),

        "warnings": "; ".join(warnings),

    })


# ============================================================
# Save validation report
# ============================================================

report = pd.DataFrame(reports)

report.to_csv(
    REPORT_FILE,
    index=False
)

# ============================================================
# Merge validation status
# ============================================================

df = df.drop(
    columns=[
        "validation_status",
        "validation_notes",
        "validation_errors",
        "validation_warnings",
    ],
    errors="ignore"
)

df = df.merge(
    report[
        [
            "scenario_id",
            "status",
            "errors",
            "warnings",
        ]
    ],
    on="scenario_id",
    how="left"
)

df = df.rename(
    columns={
        "status": "validation_status",
        "errors": "validation_errors",
        "warnings": "validation_warnings",
    }
)

df.to_csv(
    VALIDATED_FILE,
    index=False
)

# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    report["status"].value_counts()
)

print()

print(
    f"Total scenarios: {len(report)}"
)

print(
    f"Passed: "
    f"{(report['status'] == 'pass').sum()}"
)

print(
    f"Warnings: "
    f"{(report['status'] == 'warning').sum()}"
)

print(
    f"Flagged: "
    f"{(report['status'] == 'flagged').sum()}"
)

# ============================================================
# Error counts
# ============================================================

print()
print("=" * 70)
print("ERROR COUNTS")
print("=" * 70)

error_counts = (
    report[
        report["error_count"] > 0
    ]["errors"]
    .str.split("; ")
    .explode()
    .value_counts()
)

if len(error_counts) > 0:

    print(
        error_counts.to_string()
    )

else:

    print(
        "No errors found."
    )

# ============================================================
# Warning counts
# ============================================================

print()
print("=" * 70)
print("WARNING COUNTS")
print("=" * 70)

warning_counts = (
    report[
        report["warning_count"] > 0
    ]["warnings"]
    .str.split("; ")
    .explode()
    .value_counts()
)

if len(warning_counts) > 0:

    print(
        warning_counts.to_string()
    )

else:

    print(
        "No warnings found."
    )

# ============================================================
# Structural placeholder diagnostics
# ============================================================

print()
print("=" * 70)
print("PATH STRUCTURE DIAGNOSTICS")
print("=" * 70)

path1_placeholders = sum(
    is_placeholder_path(x)
    for x in df["aae_statement_path1"]
)

path2_placeholders = sum(
    is_placeholder_path(x)
    for x in df["aae_statement_path2"]
)

print(
    f"AAE PATH 1 structural placeholders: "
    f"{path1_placeholders}"
)

print(
    f"AAE PATH 2 structural placeholders: "
    f"{path2_placeholders}"
)

# ============================================================
# Output locations
# ============================================================

print()
print("Validation report:")
print(REPORT_FILE)

print()
print("Validated dataset:")
print(VALIDATED_FILE)

# ============================================================
# Next step
# ============================================================

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

if (
    path1_placeholders > 0
    or path2_placeholders > 0
):

    print(
        "IMPORTANT: Structural placeholder paths were detected."
    )

    print(
        "Inspect the conversion/parser pipeline before freezing."
    )

else:

    print(
        "No structural placeholder paths detected."
    )

if (
    (report["status"] == "flagged").sum()
    == 0
):

    print()
    print(
        "VALIDATION PASSED: No scenarios were flagged."
    )

    print(
        "The dataset is ready for manual quality review "
        "before model evaluation."
    )

else:

    print()
    print(
        "FLAGGED SCENARIOS REMAIN."
    )

    print(
        "Inspect the validation report before model evaluation."
    )

print()
print(
    "Do NOT run model evaluations until validation "
    "has been reviewed."
)
