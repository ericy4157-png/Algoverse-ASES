import pandas as pd
import numpy as np
import os
import json


# ============================================================
# Configuration
# ============================================================

ANALYSIS_ROOT = "results/analysis"

OUTPUT_FOLDER = "results/interpretation"

# Below this many paired observations, a Wilcoxon test has essentially no
# power and can produce large-looking effect sizes from noise (this is
# exactly what happened with the n=2 MoReBench rows). Rows below this
# threshold are kept in the data (never silently dropped) but flagged as
# low_power everywhere, and excluded from anything labeled a "finding".
MIN_RELIABLE_N = 10

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# Find all statistics.csv files
# ============================================================

statistics_files = []

for root, dirs, files in os.walk(ANALYSIS_ROOT):

    if "statistics.csv" in files:

        statistics_files.append(
            os.path.join(
                root,
                "statistics.csv"
            )
        )


if not statistics_files:

    raise RuntimeError(
        "No statistics.csv files found."
    )


print("Found statistics files:")

for file in statistics_files:
    print(f"  {file}")


# ============================================================
# Combine all models
# ============================================================

frames = []

for file in statistics_files:

    df = pd.read_csv(file)

    # Track provenance so any bad row can be traced back to its file,
    # since that information is lost after concat.
    df["source_file"] = file

    frames.append(df)


all_results = pd.concat(
    frames,
    ignore_index=True
)


# ============================================================
# Quarantine rows that don't fit the expected per-model,
# per-benchmark schema (NEW)
# ============================================================
# concat() will happily merge in a file that's missing model/benchmark
# columns -- e.g. a pooled/combined-across-models analysis file that
# got left in the same directory tree -- filling the gaps with NaN.
# Nothing downstream should treat those rows as a normal comparison:
# they can't be attributed to a specific model or benchmark, which
# breaks every summary and consensus calculation that groups by those
# columns. Quarantine them loudly instead of silently blending them in.

# Only structural identity columns belong here. p_value / cohens_dz are
# deliberately NOT included: they can legitimately be NaN for a genuine
# tied comparison (e.g. AAE_minus_SAE == 0 for every pair, which
# Wilcoxon can't compute a p-value for) -- that's a real result, not a
# schema problem, and must stay in the analysis.
REQUIRED_COLUMNS = ["model", "benchmark", "metric", "n"]

is_incomplete = pd.Series(False, index=all_results.index)
for col in REQUIRED_COLUMNS:
    if col not in all_results.columns:
        raise RuntimeError(
            f"Expected column '{col}' not found in any statistics.csv file."
        )
    is_incomplete |= all_results[col].isna()

quarantined = all_results[is_incomplete].copy()
all_results = all_results[~is_incomplete].copy()

# Always write (or remove) quarantined_rows.csv so it reflects the
# CURRENT run. Previously this file was only written when there was
# something to quarantine, which meant a stale quarantined_rows.csv
# from an earlier run could stick around and get mistaken for current
# output after the underlying problem was already fixed.
quarantine_path = f"{OUTPUT_FOLDER}/quarantined_rows.csv"

if len(quarantined) > 0:
    quarantined.to_csv(quarantine_path, index=False)
elif os.path.exists(quarantine_path):
    os.remove(quarantine_path)

if len(quarantined) > 0:
    print()
    print("!" * 70)
    print(
        f"WARNING: {len(quarantined)} row(s) were missing model, benchmark, "
        f"or another required field and have been EXCLUDED from all "
        f"analysis below. This usually means a statistics.csv with a "
        f"different/incompatible schema (e.g. a pooled or combined-across- "
        f"models file) exists somewhere under {ANALYSIS_ROOT} and is being "
        f"picked up alongside your per-model files."
    )
    print("Source file(s) involved:")
    for f in quarantined["source_file"].unique():
        print(f"  {f}")
    print(f"Full quarantined rows saved to: {OUTPUT_FOLDER}/quarantined_rows.csv")
    print("!" * 70)

else:
    print(
        f"\nNo quarantined rows this run "
        f"({len(all_results)} rows all have complete model/benchmark/metric/n)."
    )


# ============================================================
# Clean model names
# ============================================================

def clean_model_name(name):

    name = str(name).lower()

    if "gpt5" in name:
        return "GPT-5"

    if "deepseek" in name:
        return "DeepSeek-V4-Pro"

    if "claude" in name:
        return "Claude"

    return name


all_results["model_family"] = (
    all_results["model"]
    .apply(clean_model_name)
)


# ============================================================
# Reliability flag (NEW)
# ============================================================
# A test with n < MIN_RELIABLE_N is kept but marked. This is what was
# missing before: the n=2 MoReBench rows had no way to signal "don't
# trust this effect size" to anything downstream.

all_results["low_power"] = (
    all_results["n"] < MIN_RELIABLE_N
)


# ============================================================
# Multiple-comparisons correction (NEW)
# ============================================================
# Benjamini-Hochberg FDR correction across every test with a valid
# p-value. Not urgent while everything is null, but needed the moment
# real effects start showing up at scale.

def benjamini_hochberg(pvals):
    """Return BH-FDR adjusted p-values, same order as input. NaNs pass through."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    adjusted = np.full(n, np.nan)

    valid_mask = ~np.isnan(pvals)
    valid_idx = np.where(valid_mask)[0]
    valid_p = pvals[valid_mask]
    m = len(valid_p)

    if m == 0:
        return adjusted

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    ranks = np.arange(1, m + 1)

    bh = ranked_p * m / ranks
    # enforce monotonicity (BH step-up)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    adjusted[valid_idx[order]] = bh
    return adjusted


all_results["p_value_fdr"] = benjamini_hochberg(
    all_results["p_value"].values
)

all_results["significant_fdr05"] = (
    all_results["p_value_fdr"] < 0.05
)


# ============================================================
# Save master dataset
# ============================================================

all_results.to_csv(
    f"{OUTPUT_FOLDER}/master_statistics.csv",
    index=False
)


print()
print(
    f"Combined {len(all_results)} statistical results."
)

n_low_power = int(all_results["low_power"].sum())
print(
    f"  {n_low_power} of these have n < {MIN_RELIABLE_N} "
    f"and are flagged low_power (kept, but excluded from findings)."
)


# ============================================================
# 1. Strongest effects
# ============================================================
# CHANGED: still includes every row (nothing is hidden from the CSV),
# but now carries the low_power flag explicitly, and is sorted with
# reliable effects first so a low-n artifact can't silently sit at
# the top. p_value and n are included so magnitude is never shown
# without the context needed to judge it.

strongest_effects = (
    all_results
    .dropna(subset=["cohens_dz"])
    .assign(
        absolute_effect=lambda x:
            x["cohens_dz"].abs()
    )
    .sort_values(
        ["low_power", "absolute_effect"],
        ascending=[True, False]
    )
)

strongest_effects[
    [
        "model_family",
        "benchmark",
        "metric",
        "AAE_minus_SAE",
        "p_value",
        "cohens_dz",
        "effect_size_category",
        "n",
        "low_power"
    ]
].to_csv(
    f"{OUTPUT_FOLDER}/strongest_effects.csv",
    index=False
)


# ============================================================
# 2. Significant findings
# ============================================================

significant = all_results[
    all_results["significant_p05"] == True
].copy()

significant = significant.sort_values(
    "p_value"
)

significant.to_csv(
    f"{OUTPUT_FOLDER}/significant_findings.csv",
    index=False
)


# ============================================================
# 3. Benchmark summary
# ============================================================
# CHANGED: added min_n so a reader can see, per row, whether the
# summary is backed by n=15 or n=2 without opening master_statistics.csv.

benchmark_summary = (
    all_results
    .groupby(
        ["benchmark", "metric"]
    )
    .agg(
        mean_difference=(
            "AAE_minus_SAE",
            "mean"
        ),

        mean_effect_size=(
            "cohens_dz",
            "mean"
        ),

        models_tested=(
            "model_family",
            "nunique"
        ),

        significant_models=(
            "significant_p05",
            "sum"
        ),

        min_n=(
            "n",
            "min"
        ),

        any_low_power=(
            "low_power",
            "any"
        )
    )
    .reset_index()
)


benchmark_summary.to_csv(
    f"{OUTPUT_FOLDER}/benchmark_summary.csv",
    index=False
)


# ============================================================
# 4. Model summary
# ============================================================

model_summary = (
    all_results
    .groupby(
        ["model_family", "benchmark"]
    )
    .agg(

        mean_difference=(
            "AAE_minus_SAE",
            "mean"
        ),

        mean_effect_size=(
            "cohens_dz",
            "mean"
        ),

        significant_results=(
            "significant_p05",
            "sum"
        ),

        total_results=(
            "metric",
            "count"
        ),

        min_n=(
            "n",
            "min"
        )
    )
    .reset_index()
)


model_summary[
    "significance_rate"
] = (
    model_summary["significant_results"]
    /
    model_summary["total_results"]
)


model_summary.to_csv(
    f"{OUTPUT_FOLDER}/model_summary.csv",
    index=False
)


# ============================================================
# 5. Cross-model agreement
# ============================================================
# CHANGED: added min_n_across_models and a reliability caveat, since
# a "2/3 models agree" vote means something very different if one of
# those models' tests had n=2 vs. n=15. Also: with only 3 models,
# 2/3 agreement is roughly what you'd expect from chance alone
# (~75% under a fair coin-flip null), so agreement_rate is only
# meaningful once you have considerably more models.

MIN_MODELS_FOR_RELIABLE_CONSENSUS = 5

agreement_rows = []


for benchmark in all_results["benchmark"].unique():

    benchmark_df = all_results[
        all_results["benchmark"] == benchmark
    ]

    for metric in benchmark_df["metric"].unique():

        subset = benchmark_df[
            benchmark_df["metric"] == metric
        ].dropna(
            subset=["AAE_minus_SAE"]
        )

        if len(subset) == 0:
            continue

        positive = (
            subset["AAE_minus_SAE"] > 0
        ).sum()

        negative = (
            subset["AAE_minus_SAE"] < 0
        ).sum()

        zero = (
            subset["AAE_minus_SAE"] == 0
        ).sum()

        total = len(subset)

        if negative > positive and negative > zero:
            consensus_direction = "AAE_lower"

        elif positive > negative and positive > zero:
            consensus_direction = "AAE_higher"

        else:
            consensus_direction = "mixed"

        agreement_rows.append({

            "benchmark": benchmark,

            "metric": metric,

            "models": total,

            "AAE_lower_models": negative,

            "AAE_higher_models": positive,

            "no_difference_models": zero,

            "consensus_direction":
                consensus_direction,

            "agreement_rate":
                round(
                    max(
                        positive,
                        negative,
                        zero
                    ) / total,
                    3
                ),

            "min_n_across_models":
                int(subset["n"].min()),

            "reliable_consensus":
                total >= MIN_MODELS_FOR_RELIABLE_CONSENSUS
                and subset["n"].min() >= MIN_RELIABLE_N
        })


agreement_df = pd.DataFrame(
    agreement_rows
)


agreement_df.to_csv(
    f"{OUTPUT_FOLDER}/cross_model_agreement.csv",
    index=False
)


# ============================================================
# 6. Generate machine-readable findings
# ============================================================
# CHANGED: findings.json now only draws from low_power == False rows.
# This is the fix for the specific bug that surfaced: the n=2 MoReBench
# / p=1.0 row could no longer become the reported "strongest_effect"
# even though it still exists, visibly flagged, in strongest_effects.csv.

findings = []

reliable_effects = strongest_effects[
    strongest_effects["low_power"] == False
]

# ------------------------------------------------------------
# Strongest overall RELIABLE effect
# ------------------------------------------------------------

if len(reliable_effects) > 0:

    strongest = reliable_effects.iloc[0]

    findings.append({

        "finding_type":
            "strongest_effect",

        "model":
            strongest["model_family"],

        "benchmark":
            strongest["benchmark"],

        "metric":
            strongest["metric"],

        "effect":
            float(strongest["cohens_dz"]),

        "difference":
            float(strongest["AAE_minus_SAE"]),

        "p_value":
            (
                float(strongest["p_value"])
                if pd.notna(
                    strongest["p_value"]
                )
                else None
            ),

        "n":
            int(strongest["n"]),

        "significant_p05":
            bool(strongest["significant_p05"])
            if pd.notna(strongest["significant_p05"])
            else False,

        "interpretation":
            "Largest observed standardized dialect difference among "
            f"tests with n >= {MIN_RELIABLE_N} (unreliable low-n effects "
            "are excluded from this ranking; see strongest_effects.csv "
            "for the full list with low_power flags)."
    })

else:
    print(
        f"\nWARNING: no comparisons with n >= {MIN_RELIABLE_N} were "
        f"found. No 'strongest_effect' finding was generated — all "
        f"observed effects are currently underpowered."
    )


# ------------------------------------------------------------
# Significant effects
# ------------------------------------------------------------
# CHANGED: now carries n, low_power, and the FDR-corrected result
# alongside the raw p-value, so the report-writing LLM can see
# whether a "significant" result also survives multiple-comparisons
# correction and whether it's backed by enough data.

for _, row in significant.iterrows():

    findings.append({

        "finding_type":
            "statistically_significant",

        "model":
            row["model_family"],

        "benchmark":
            row["benchmark"],

        "metric":
            row["metric"],

        "effect":
            (
                float(row["cohens_dz"])
                if pd.notna(row["cohens_dz"])
                else None
            ),

        "difference":
            float(row["AAE_minus_SAE"]),

        "p_value":
            float(row["p_value"]),

        "p_value_fdr":
            (
                float(row["p_value_fdr"])
                if pd.notna(row["p_value_fdr"])
                else None
            ),

        "significant_after_fdr":
            bool(row["significant_fdr05"]),

        "n":
            int(row["n"]),

        "low_power":
            bool(row["low_power"]),

        "interpretation":
            (
                "AAE received lower ratings than SAE."
                if row["AAE_minus_SAE"] < 0
                else
                "AAE received higher ratings than SAE."
            )
    })


# ============================================================
# Save findings JSON
# ============================================================

with open(
    f"{OUTPUT_FOLDER}/findings.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        findings,
        f,
        indent=2
    )


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 70)
print("INTERPRETIVE SUMMARY")
print("=" * 70)


print()
print("Models:")
print(
    all_results["model_family"]
    .unique()
)


print()
print("Benchmarks:")
print(
    all_results["benchmark"]
    .unique()
)


print()
print(
    f"Total statistical comparisons: "
    f"{len(all_results)}"
)


print(
    f"Statistically significant results (uncorrected): "
    f"{len(significant)}"
)

print(
    f"Statistically significant results (BH-FDR corrected): "
    f"{int(all_results['significant_fdr05'].sum())}"
)

print(
    f"Comparisons flagged low_power (n < {MIN_RELIABLE_N}): "
    f"{n_low_power} / {len(all_results)}"
)


print()
print("Strongest RELIABLE effects (n >= {}):".format(MIN_RELIABLE_N))

print(
    reliable_effects[
        [
            "model_family",
            "benchmark",
            "metric",
            "AAE_minus_SAE",
            "p_value",
            "cohens_dz",
            "n"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

low_power_effects = strongest_effects[
    strongest_effects["low_power"] == True
]

if len(low_power_effects) > 0:
    print()
    print(
        f"NOTE: {len(low_power_effects)} additional comparisons had "
        f"n < {MIN_RELIABLE_N} and were excluded from the ranking above. "
        f"They remain in strongest_effects.csv, flagged low_power=True, "
        f"for transparency — do not report their effect sizes as findings."
    )

print()
print("Cross-model agreement:")

print(
    agreement_df.to_string(index=False)
)

n_unreliable_consensus = int((~agreement_df["reliable_consensus"]).sum())
if n_unreliable_consensus > 0:
    print(
        f"\nNOTE: {n_unreliable_consensus} of {len(agreement_df)} "
        f"agreement rows do not meet the reliability bar "
        f"(>= {MIN_MODELS_FOR_RELIABLE_CONSENSUS} models, each with "
        f"n >= {MIN_RELIABLE_N}). With only {all_results['model_family'].nunique()} "
        f"model(s) currently, treat 'consensus_direction' as descriptive, "
        f"not as statistical agreement."
    )


print()
print("=" * 70)
print("Interpretation complete.")
print("=" * 70)

print()
print(
    f"Results saved to: {OUTPUT_FOLDER}/"
)