import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/full/candidate_scenarios.csv"
OUTPUT_FILE = "data/full/master_scenarios.csv"

TARGET_PER_SOURCE = 125
RANDOM_SEED = 20260821

MIN_SCENARIO_LENGTH = 200
MAX_SCENARIO_LENGTH = 1800

os.makedirs(
    "data/full",
    exist_ok=True
)


# ============================================================
# Load candidate pool
# ============================================================

print("=" * 70)
print("LOADING CANDIDATE DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} candidate scenarios.")


# ============================================================
# Basic validation
# ============================================================

required_columns = [
    "scenario_id",
    "source",
    "source_id",
    "scenario_text",
    "path1",
    "path2"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Clean text
# ============================================================

for column in [
    "scenario_text",
    "path1",
    "path2"
]:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# Quality filters
# ============================================================

df["scenario_length"] = (
    df["scenario_text"].str.len()
)

df["valid_structure"] = (
    (df["scenario_text"].str.len() > 0)
    &
    (df["path1"].str.len() > 0)
    &
    (df["path2"].str.len() > 0)
)

df["valid_length"] = (
    (df["scenario_length"] >= MIN_SCENARIO_LENGTH)
    &
    (df["scenario_length"] <= MAX_SCENARIO_LENGTH)
)

before = len(df)

df = df[
    df["valid_structure"]
    &
    df["valid_length"]
].copy()

print()
print("=" * 70)
print("QUALITY FILTERING")
print("=" * 70)

print(f"Before filtering: {before}")
print(f"After filtering:  {len(df)}")
print(f"Removed:          {before - len(df)}")


# ============================================================
# Check source availability
# ============================================================

print()
print("Available scenarios by source:")

print(
    df["source"].value_counts()
)


for source in [
    "Daily_Dilemmas",
    "MoReBench"
]:

    count = int(
        (df["source"] == source).sum()
    )

    if count < TARGET_PER_SOURCE:

        raise RuntimeError(
            f"Not enough {source} scenarios after filtering: "
            f"{count} available, "
            f"{TARGET_PER_SOURCE} required."
        )


# ============================================================
# Stratified sampling
# ============================================================

print()
print("=" * 70)
print("SELECTING MASTER DATASET")
print("=" * 70)

selected = []

for source in [
    "Daily_Dilemmas",
    "MoReBench"
]:

    source_df = df[
        df["source"] == source
    ].copy()

    sample = source_df.sample(
        n=TARGET_PER_SOURCE,
        random_state=RANDOM_SEED
    )

    selected.append(sample)

    print(
        f"{source}: selected {len(sample)}"
    )


master = pd.concat(
    selected,
    ignore_index=True
)


# ============================================================
# Shuffle final dataset
# ============================================================

master = master.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# Assign FINAL scenario IDs
# ============================================================

master["scenario_id"] = (
    range(1, len(master) + 1)
)


# ============================================================
# Add dataset version
# ============================================================

master["dataset_version"] = "v1.0"

master["selection_seed"] = RANDOM_SEED


# ============================================================
# Final validation
# ============================================================

expected_total = (
    TARGET_PER_SOURCE * 2
)

if len(master) != expected_total:

    raise RuntimeError(
        f"Expected {expected_total} scenarios, "
        f"got {len(master)}."
    )


print()
print("=" * 70)
print("FINAL MASTER DATASET")
print("=" * 70)

print(
    f"Total scenarios: {len(master)}"
)

print()
print("By source:")

print(
    master["source"].value_counts()
)


print()
print("Scenario length:")

print(
    master["scenario_text"]
    .str.len()
    .describe()
)


# ============================================================
# Save
# ============================================================

columns = [
    "scenario_id",
    "source",
    "source_id",
    "scenario_text",
    "path1",
    "path2"
]

for column in [
    "rubric",
    "dilemma_type",
    "theory",
    "role_domain"
]:

    if column in master.columns:

        columns.append(column)


columns.extend([
    "dataset_version",
    "selection_seed"
])


master[
    columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Saved master dataset to:\n{OUTPUT_FILE}"
)

print()
print(
    "This dataset is now the frozen scenario pool "
    "for the full experiment."
)