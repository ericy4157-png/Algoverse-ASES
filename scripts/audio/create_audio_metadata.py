import os
import re
import pandas as pd

AUDIO_DIR = "data/audio"
OUTPUT_FILE = "audio/metadata/audio_metadata.csv"

rows = []

for dialect in ["sae", "aae"]:
    dialect_dir = os.path.join(AUDIO_DIR, dialect)

    for filename in os.listdir(dialect_dir):
        if not filename.endswith(".wav"):
            continue

        # Example:
        # scenario13_AAE_path1.wav

        match = re.match(
            r"scenario(\d+)_(AAE|SAE)_path(\d+)\.wav",
            filename,
            re.IGNORECASE
        )

        if not match:
            print(f"WARNING: Could not parse {filename}")
            continue

        scenario_id = int(match.group(1))
        dialect_from_filename = match.group(2).upper()
        path = int(match.group(3))

        # Store relative path
        audio_file = os.path.join(
            AUDIO_DIR,
            dialect,
            filename
        )

        rows.append({
            "scenario_id": scenario_id,
            "dialect": dialect_from_filename,
            "path": path,
            "audio_file": audio_file
        })

df = pd.DataFrame(rows)

# Sort for reproducibility
df = df.sort_values(
    ["scenario_id", "dialect", "path"]
).reset_index(drop=True)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 60)
print("Audio metadata created")
print("=" * 60)

print(f"Total audio files: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")
print()

print("By dialect:")
print(df["dialect"].value_counts())

print()
print("By path:")
print(df["path"].value_counts())

print()
print("First 10 rows:")
print(df.head(10).to_string(index=False))