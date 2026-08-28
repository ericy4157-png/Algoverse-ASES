import pandas as pd
import os

# ============================================================
# Configuration
# ============================================================

AUDIO_METADATA = "audio/metadata/audio_metadata.csv"

SOURCE_FILE = "data/7_21 Compiled Pilot AAVE Conversions - Sheet1 (1).csv"

OUTPUT_FILE = "audio/metadata/audio_dataset.csv"


# ============================================================
# Load audio metadata
# ============================================================

audio = pd.read_csv(AUDIO_METADATA)

# ============================================================
# Load original scenario source
# ============================================================

source = pd.read_csv(SOURCE_FILE)

# Remove the completely empty/category rows
source = source[source["original_sae"].notna()].copy()

# Reset index so scenario IDs are 1–20
source = source.reset_index(drop=True)

source["scenario_id"] = source.index + 1


# ============================================================
# Assign benchmark
# ============================================================

# Scenarios 1–10 = Daily Dilemmas
# Scenarios 11–20 = MoReBench

source["benchmark"] = source["scenario_id"].apply(
    lambda x: "Daily_Dilemmas" if x <= 10 else "MoReBench"
)


# ============================================================
# Convert source into long format
# ============================================================

rows = []

for _, row in source.iterrows():

    scenario_id = row["scenario_id"]
    benchmark = row["benchmark"]

    for dialect in ["SAE", "AAE"]:

        for path in [1, 2]:

            if dialect == "SAE":
                text = row[f"sae_path{path}"]
            else:
                text = row[f"aae_path{path}"]

            rows.append({
                "scenario_id": scenario_id,
                "benchmark": benchmark,
                "dialect": dialect,
                "path": path,
                "text": text
            })


text_metadata = pd.DataFrame(rows)


# ============================================================
# Merge with audio files
# ============================================================

dataset = audio.merge(
    text_metadata,
    on=["scenario_id", "dialect", "path"],
    how="left"
)


# ============================================================
# Validate
# ============================================================

print("=" * 60)
print("Audio dataset")
print("=" * 60)

print(f"Audio rows: {len(audio)}")
print(f"Dataset rows: {len(dataset)}")

print()
print("Missing benchmark:", dataset["benchmark"].isna().sum())
print("Missing text:", dataset["text"].isna().sum())
print("Missing audio:", dataset["audio_file"].isna().sum())

print()
print("Benchmark counts:")
print(dataset["benchmark"].value_counts())

print()
print("Dialect counts:")
print(dataset["dialect"].value_counts())

print()
print("Path counts:")
print(dataset["path"].value_counts())


# ============================================================
# Stop if anything is missing
# ============================================================

if dataset["benchmark"].isna().any():
    raise RuntimeError("ERROR: Some rows are missing benchmark.")

if dataset["text"].isna().any():
    raise RuntimeError("ERROR: Some rows are missing text.")

if dataset["audio_file"].isna().any():
    raise RuntimeError("ERROR: Some rows are missing audio files.")


# ============================================================
# Save
# ============================================================

os.makedirs("audio/metadata", exist_ok=True)

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)
print(f"Saved to: {OUTPUT_FILE}")

print()
print("First 10 rows:")
print(
    dataset.head(10).to_string(index=False)
)