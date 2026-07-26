import pandas as pd

# Load current dataset
df = pd.read_csv("data/multivalue_outputs.csv")

# ----- CHANGE THESE NUMBERS IF YOUR SPLIT IS DIFFERENT -----
# Example:
# first 10 scenarios = Daily Dilemmas
# next 10 scenarios = MoReBench

def get_benchmark(scenario_id):
    if scenario_id <= 10:
        return "Daily_Dilemmas"
    else:
        return "MoReBench"

df["benchmark"] = df["scenario_id"].apply(get_benchmark)

# Reorder columns
cols = [
    "scenario_id",
    "benchmark",
    "dialect",
    "path",
    "text"
]

df = df[cols]

# Save
df.to_csv("data/multivalue_outputs.csv", index=False)

print("Benchmark column added.")
print(df.head())