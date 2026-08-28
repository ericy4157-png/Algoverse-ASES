import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

MASTER_FILE = "data/full/master_scenarios.csv"

# Existing pilot conversions
PILOT_CONVERSION_FILE = (
    "data/7_21 Compiled Pilot AAVE Conversions - Sheet1 (1).csv"
)

OUTPUT_DIR = "data/full"
OUTPUT_FILE = f"{OUTPUT_DIR}/experiment_manifest.csv"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load master dataset
# ============================================================

print("=" * 70)
print("LOADING MASTER DATASET")
print("=" * 70)

master = pd.read_csv(MASTER_FILE)

print(f"Loaded {len(master)} scenarios.")


# ============================================================
# Validate master dataset
# ============================================================

required = [
    "scenario_id",
    "source",
    "scenario_text",
    "path1",
    "path2"
]

missing = [
    c for c in required
    if c not in master.columns
]

if missing:
    raise RuntimeError(
        f"Master dataset missing columns: {missing}"
    )

if len(master) != 250:
    raise RuntimeError(
        f"Expected 250 scenarios, found {len(master)}."
    )


# ============================================================
# Load existing pilot conversions
# ============================================================

print()
print("=" * 70)
print("LOADING EXISTING AAE CONVERSIONS")
print("=" * 70)

if os.path.exists(PILOT_CONVERSION_FILE):

    conversions = pd.read_csv(
        PILOT_CONVERSION_FILE
    )

    print(
        f"Loaded {len(conversions)} existing "
        "conversion records."
    )

    print(
        "Columns:",
        list(conversions.columns)
    )

else:

    conversions = None

    print(
        "No existing conversion file found."
    )


# ============================================================
# Build SAE rows
# ============================================================

rows = []

for _, scenario in master.iterrows():

    scenario_id = int(
        scenario["scenario_id"]
    )

    source = scenario["source"]

    scenario_text = str(
        scenario["scenario_text"]
    ).strip()

    path1 = str(
        scenario["path1"]
    ).strip()

    path2 = str(
        scenario["path2"]
    ).strip()


    # --------------------------------------------------------
    # SAE Path 1
    # --------------------------------------------------------

    rows.append({

        "scenario_id": scenario_id,

        "source": source,

        "dialect": "SAE",

        "path": 1,

        "text": scenario_text,

        "action": path1,

        "conversion_status": "original",

    })


    # --------------------------------------------------------
    # SAE Path 2
    # --------------------------------------------------------

    rows.append({

        "scenario_id": scenario_id,

        "source": source,

        "dialect": "SAE",

        "path": 2,

        "text": scenario_text,

        "action": path2,

        "conversion_status": "original",

    })


# ============================================================
# AAE rows
# ============================================================

print()
print("=" * 70)
print("BUILDING AAE ROWS")
print("=" * 70)

print(
    "AAE conversions are NOT automatically generated here."
)

print(
    "Rows will be marked conversion_required until "
    "validated AAE versions are supplied."
)


for _, scenario in master.iterrows():

    scenario_id = int(
        scenario["scenario_id"]
    )

    source = scenario["source"]

    scenario_text = str(
        scenario["scenario_text"]
    ).strip()

    path1 = str(
        scenario["path1"]
    ).strip()

    path2 = str(
        scenario["path2"]
    ).strip()


    # --------------------------------------------------------
    # AAE Path 1
    # --------------------------------------------------------

    rows.append({

        "scenario_id": scenario_id,

        "source": source,

        "dialect": "AAE",

        "path": 1,

        "text": scenario_text,

        "action": path1,

        "conversion_status":
            "conversion_required",

    })


    # --------------------------------------------------------
    # AAE Path 2
    # --------------------------------------------------------

    rows.append({

        "scenario_id": scenario_id,

        "source": source,

        "dialect": "AAE",

        "path": 2,

        "text": scenario_text,

        "action": path2,

        "conversion_status":
            "conversion_required",

    })


# ============================================================
# Create manifest
# ============================================================

manifest = pd.DataFrame(rows)


# ============================================================
# Sort deterministically
# ============================================================

manifest = manifest.sort_values(
    [
        "scenario_id",
        "dialect",
        "path"
    ]
).reset_index(drop=True)


# ============================================================
# Add stable evaluation IDs
# ============================================================

manifest.insert(
    0,
    "evaluation_id",
    [
        f"E{i:04d}"
        for i in range(
            1,
            len(manifest) + 1
        )
    ]
)


# ============================================================
# Validate
# ============================================================

print()
print("=" * 70)
print("MANIFEST VALIDATION")
print("=" * 70)

print(
    f"Total rows: {len(manifest)}"
)

print()
print("By dialect:")

print(
    manifest["dialect"]
    .value_counts()
)

print()
print("By path:")

print(
    manifest["path"]
    .value_counts()
)

print()
print("By source:")

print(
    manifest["source"]
    .value_counts()
)

print()
print("By conversion status:")

print(
    manifest["conversion_status"]
    .value_counts()
)


# Expected structure

if len(manifest) != 1000:

    raise RuntimeError(
        f"Expected exactly 1,000 evaluations, "
        f"got {len(manifest)}."
    )


for dialect in [
    "SAE",
    "AAE"
]:

    count = (
        manifest["dialect"] == dialect
    ).sum()

    if count != 500:

        raise RuntimeError(
            f"Expected 500 {dialect} rows, "
            f"got {count}."
        )


for path in [
    1,
    2
]:

    count = (
        manifest["path"] == path
    ).sum()

    if count != 500:

        raise RuntimeError(
            f"Expected 500 path {path} rows, "
            f"got {count}."
        )


# ============================================================
# Save
# ============================================================

manifest.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Saved manifest to:\n{OUTPUT_FILE}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "AAE rows are currently marked "
    "'conversion_required'."
)

print(
    "Do NOT run model evaluations yet."
)