import os

import pandas as pd
from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "data/full"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load Hugging Face dataset
# ============================================================

def load_hf_dataset(name, config=None):

    print()
    print("=" * 70)
    print(f"Loading: {name}")

    if config:
        print(f"Config: {config}")

    print("=" * 70)

    if config:
        dataset = load_dataset(name, config)
    else:
        dataset = load_dataset(name)

    frames = []

    for split_name, split in dataset.items():

        df = split.to_pandas()

        df["original_split"] = split_name

        frames.append(df)

        print(
            f"  {split_name}: {len(df)} rows"
        )

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# DailyDilemmas
# ============================================================

daily = load_hf_dataset(
    "kellycyy/DailyDilemmas",
    "Dilemmas_with_values_aggregated"
)

print()
print("DailyDilemmas columns:")
print(list(daily.columns))


# ============================================================
# MoReBench
# ============================================================

more = load_hf_dataset(
    "morebench/morebench"
)

print()
print("MoReBench columns:")
print(list(more.columns))


# ============================================================
# Normalize DailyDilemmas
# ============================================================

def normalize_daily(df):

    print()
    print("=" * 70)
    print("NORMALIZING DAILYDILEMMAS")
    print("=" * 70)

    required = [
        "dilemma_idx",
        "basic_situation",
        "dilemma_situation",
        "action",
        "negative_consequence"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"DailyDilemmas missing columns: {missing}"
        )

    # DailyDilemmas stores individual actions rather than
    # explicit path1/path2 columns.
    #
    # Therefore, group actions belonging to the same dilemma
    # and use the first two genuine actions as the two paths.

    rows = []

    grouped = df.groupby("dilemma_idx")

    for dilemma_id, group in grouped:

        group = group.reset_index(drop=True)

        # We need at least two genuine alternatives.
        if len(group) < 2:
            continue

        situation = str(
            group.loc[0, "dilemma_situation"]
        ).strip()

        basic = str(
            group.loc[0, "basic_situation"]
        ).strip()

        actions = []

        for _, row in group.iterrows():

            action = str(
                row["action"]
            ).strip()

            consequence = str(
                row["negative_consequence"]
            ).strip()

            if not action or action == "nan":
                continue

            if consequence and consequence != "nan":

                action_text = (
                    f"{action}\n"
                    f"Consequence: {consequence}"
                )

            else:

                action_text = action

            actions.append(action_text)

        # We require exactly two usable alternatives.
        if len(actions) < 2:
            continue

        path1 = actions[0]
        path2 = actions[1]

        # Construct the scenario text.
        scenario_text = situation

        if basic and basic != "nan":

            scenario_text = (
                f"{basic}\n\n"
                f"{situation}"
            )

        rows.append({

            "source": "Daily_Dilemmas",

            "source_id": str(dilemma_id),

            "scenario_text": scenario_text,

            "path1": path1,

            "path2": path2

        })

    result = pd.DataFrame(rows)

    print(
        f"Created {len(result)} DailyDilemmas scenarios."
    )

    return result


# ============================================================
# Normalize MoReBench
# ============================================================

def normalize_more(df):

    print()
    print("=" * 70)
    print("NORMALIZING MOREBENCH")
    print("=" * 70)

    required = [
        "DILEMMA",
        "CONTEXT",
        "RUBRIC"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"MoReBench missing columns: {missing}"
        )

    # IMPORTANT:
    #
    # The current MoReBench dataset does NOT contain explicit
    # path1/path2 alternatives.
    #
    # It contains a dilemma, context, and rubric. Therefore,
    # we must NOT invent OPTION_A / OPTION_B.
    #
    # Since this master dataset is specifically for a
    # matched-choice dialect experiment, MoReBench is excluded
    # rather than creating artificial alternatives.

    print(
        "MoReBench does not contain explicit path1/path2 "
        "alternatives."
    )

    print(
        "MoReBench will therefore NOT be included in the "
        "paired-choice master dataset."
    )

    print(
        f"Skipping {len(df)} MoReBench rows."
    )

    return pd.DataFrame(
        columns=[
            "source",
            "source_id",
            "scenario_text",
            "path1",
            "path2",
            "rubric",
            "dilemma_type",
            "theory",
            "role_domain"
        ]
    )


# ============================================================
# Normalize datasets
# ============================================================

daily_normalized = normalize_daily(
    daily
)

more_normalized = normalize_more(
    more
)


# ============================================================
# Combine
# ============================================================

combined = pd.concat(
    [
        daily_normalized,
        more_normalized
    ],
    ignore_index=True,
    sort=False
)


# ============================================================
# Clean text
# ============================================================

for column in [
    "scenario_text",
    "path1",
    "path2"
]:

    if column in combined.columns:

        combined[column] = (
            combined[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )


# ============================================================
# Remove empty scenarios
# ============================================================

combined = combined[
    combined["scenario_text"].str.len() > 0
].copy()


# ============================================================
# Remove scenarios without two genuine paths
# ============================================================

before_paths = len(combined)

combined = combined[
    (combined["path1"].str.len() > 0)
    & (combined["path2"].str.len() > 0)
    & (
        ~combined["path1"]
        .str.fullmatch(
            r"OPTION[_ ]?A",
            case=False,
            na=False
        )
    )
    & (
        ~combined["path2"]
        .str.fullmatch(
            r"OPTION[_ ]?B",
            case=False,
            na=False
        )
    )
].copy()

removed_paths = before_paths - len(combined)

print()

print(
    f"Removed {removed_paths} scenarios without "
    "genuine paired alternatives."
)


# ============================================================
# Deduplicate
# ============================================================

before = len(combined)

combined["dedupe_text"] = (
    combined["scenario_text"]
    .str.lower()
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)

combined = combined.drop_duplicates(
    subset=["dedupe_text"]
).copy()

removed = before - len(combined)

print()

print(
    f"Removed {removed} duplicate scenarios."
)


# ============================================================
# Assign master IDs
# ============================================================

combined.insert(
    0,
    "scenario_id",
    range(
        1,
        len(combined) + 1
    )
)


# ============================================================
# Summary
# ============================================================

print()

print("=" * 70)
print("MASTER CANDIDATE DATASET")
print("=" * 70)

print(
    f"Total scenarios: {len(combined)}"
)

print()

print("By source:")

print(
    combined["source"].value_counts()
)


# ============================================================
# Verify no placeholders remain
# ============================================================

placeholder_a = (
    combined["path1"]
    .str.fullmatch(
        r"OPTION[_ ]?A",
        case=False,
        na=False
    )
    .sum()
)

placeholder_b = (
    combined["path2"]
    .str.fullmatch(
        r"OPTION[_ ]?B",
        case=False,
        na=False
    )
    .sum()
)

print()

print("Placeholder validation:")

print(
    f"OPTION_A in path1: {placeholder_a}"
)

print(
    f"OPTION_B in path2: {placeholder_b}"
)

if placeholder_a > 0 or placeholder_b > 0:

    raise RuntimeError(
        "Placeholder alternatives detected. "
        "The master dataset must contain genuine choices."
    )


# ============================================================
# Save
# ============================================================

output_file = (
    f"{OUTPUT_DIR}/candidate_scenarios.csv"
)

columns = [
    "scenario_id",
    "source",
    "source_id",
    "scenario_text",
    "path1",
    "path2"
]


# Keep optional metadata if available.

for column in [
    "rubric",
    "dilemma_type",
    "theory",
    "role_domain"
]:

    if column in combined.columns:

        columns.append(column)


combined[
    columns
].to_csv(
    output_file,
    index=False
)


print()

print("Saved:")

print(
    output_file
)


print()

print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "Do NOT run the model evaluation yet."
)

print(
    "First inspect candidate_scenarios.csv."
)

print(
    "Verify that every path1/path2 contains genuine "
    "alternative text and that no OPTION_A/OPTION_B "
    "placeholders remain."
)