import pandas as pd

input_file = "data/7_21 Compiled Pilot AAVE Conversions - Sheet1 (1).csv"

df = pd.read_csv(input_file)

df = df.dropna(subset=["original_sae"])

rows = []

for idx, row in df.iterrows():
    scenario_id = idx + 1

    versions = [
        ("SAE", 1, row["sae_path1"]),
        ("AAE", 1, row["aae_path1"]),
        ("SAE", 2, row["sae_path2"]),
        ("AAE", 2, row["aae_path2"]),
    ]

    for dialect, path, text in versions:
        if pd.notna(text):
            rows.append({
                "scenario_id": scenario_id,
                "dialect": dialect,
                "path": path,
                "text": text
            })

output_df = pd.DataFrame(rows)

output_file = "data/multivalue_outputs.csv"
output_df.to_csv(output_file, index=False)

print(f"Created {output_file}")
print(output_df.head())
print(f"Total rows: {len(output_df)}")