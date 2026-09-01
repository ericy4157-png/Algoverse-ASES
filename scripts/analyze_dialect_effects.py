#!/usr/bin/env python3

"""
Dialect Bias Analysis
=====================

Analyzes the four completed TEXT evaluation datasets:

    Qwen3-30B-A3B-Instruct-2507
    GPT-5
    DeepSeek-V4-Pro
    Claude Opus 5

Design:
    394 scenarios
    3 dialects: SAE, AAE, ChE
    2 arms: A, B

Each comparison is paired within:
    parent_id × arm

Primary contrasts:
    AAE - SAE
    ChE  - SAE

Outcomes:
    moral_acceptability
    responsibility
    trustworthiness
    compassion
    fairness
    consequences
    recommendation

Outputs:
    results/analysis/
        paired_effects_all.csv
        model_summary.csv
        forest_plot_all_outcomes.png
        forest_plot_moral_acceptability.png
        forest_plot_recommendation.png
        validation_report.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_FILES = {
    "Qwen3-30B-A3B-Instruct-2507":
        "results/full/qwen3-30b-a3b-instruct-2507_stimuli_all_dialects.csv",

    "GPT-5":
        "results/full/gpt-5_stimuli_all_dialects.csv",

    "DeepSeek-V4-Pro":
        "results/full/deepseek_v4_pro_stimuli_all_dialects.csv",

    "Claude Opus 5":
        "results/full/claude_opus_5_stimuli_all_dialects.csv",
}


OUTCOME_COLUMNS = [
    "moral_acceptability",
    "responsibility",
    "trustworthiness",
    "compassion",
    "fairness",
    "consequences",
    "recommendation",
]

DIALECTS = ["SAE", "AAE", "ChE"]

EXPECTED_ROWS = 2364
EXPECTED_SCENARIOS = 394
EXPECTED_ROWS_PER_DIALECT = 788

OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def paired_effect(x):
    """
    Calculate paired mean difference and 95% CI.

    x = dialect score - SAE score

    Returns:
        estimate
        standard_error
        ci_low
        ci_high
        t_stat
        p_value
        n
    """

    x = pd.to_numeric(x, errors="coerce").dropna()

    n = len(x)

    if n == 0:
        return {
            "estimate": np.nan,
            "standard_error": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n": 0,
        }

    mean = x.mean()

    if n == 1:
        return {
            "estimate": mean,
            "standard_error": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n": 1,
        }

    sd = x.std(ddof=1)
    se = sd / np.sqrt(n)

    t_stat = mean / se if se != 0 else np.nan

    # scipy is used only for the t distribution
    from scipy import stats

    p_value = (
        2 * stats.t.sf(abs(t_stat), df=n - 1)
        if np.isfinite(t_stat)
        else 0.0 if mean != 0 else 1.0
    )

    critical = stats.t.ppf(0.975, df=n - 1)

    ci_low = mean - critical * se
    ci_high = mean + critical * se

    return {
        "estimate": mean,
        "standard_error": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "p_value": p_value,
        "n": n,
    }


def bonferroni(p_values):
    """
    Bonferroni correction.
    """
    p_values = np.asarray(p_values, dtype=float)

    m = np.sum(np.isfinite(p_values))

    if m == 0:
        return np.full_like(p_values, np.nan)

    return np.minimum(p_values * m, 1.0)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DIALECT BIAS ANALYSIS")
print("=" * 80)

all_effects = []
validation_lines = []

for model_name, filepath in RESULT_FILES.items():

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find result file:\n{filepath}"
        )

    df = pd.read_csv(path)

    print(f"Loaded rows: {len(df):,}")

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [str(c).strip() for c in df.columns]

    # --------------------------------------------------------
    # Determine scenario column
    # --------------------------------------------------------

    if "parent_id" in df.columns:
        scenario_col = "parent_id"

    elif "scenario_id" in df.columns:
        scenario_col = "scenario_id"

    else:
        raise ValueError(
            f"{model_name}: No parent_id or scenario_id column found."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        scenario_col,
        "item_id",
        "dialect",
        "arm",
        *OUTCOME_COLUMNS,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name}: Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if len(df) != EXPECTED_ROWS:
        validation_lines.append(
            f"{model_name}: WARNING - expected "
            f"{EXPECTED_ROWS} rows but found {len(df)}"
        )
    else:
        validation_lines.append(
            f"{model_name}: PASS - {len(df)} rows"
        )

    n_scenarios = df[scenario_col].nunique()

    print(f"Scenarios: {n_scenarios}")
    print(
        "Dialects:",
        sorted(df["dialect"].dropna().unique())
    )

    print(
        "Arms:",
        sorted(df["arm"].dropna().unique())
    )

    if n_scenarios != EXPECTED_SCENARIOS:
        validation_lines.append(
            f"{model_name}: WARNING - expected "
            f"{EXPECTED_SCENARIOS} scenarios but found "
            f"{n_scenarios}"
        )

    # --------------------------------------------------------
    # Check dialect × arm balance
    # --------------------------------------------------------

    counts = pd.crosstab(
        df["dialect"],
        df["arm"]
    )

    print("\nDialect × arm counts:")
    print(counts)

    # --------------------------------------------------------
    # Check duplicates
    # --------------------------------------------------------

    duplicate_keys = df.duplicated(
        subset=[scenario_col, "dialect", "arm"],
        keep=False
    )

    n_duplicate_rows = duplicate_keys.sum()

    print(
        f"\nDuplicate scenario/dialect/arm rows: "
        f"{n_duplicate_rows}"
    )

    if n_duplicate_rows:
        validation_lines.append(
            f"{model_name}: WARNING - "
            f"{n_duplicate_rows} duplicate rows"
        )
    else:
        validation_lines.append(
            f"{model_name}: PASS - no duplicate "
            f"scenario/dialect/arm rows"
        )

    # --------------------------------------------------------
    # Evaluation status
    # --------------------------------------------------------

    if "evaluation_status" in df.columns:

        print("\nEvaluation status:")
        print(df["evaluation_status"].value_counts(dropna=False))

    # --------------------------------------------------------
    # Convert outcomes to numeric
    # --------------------------------------------------------

    for col in OUTCOME_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Build paired dataset
    #
    # One observation = scenario × arm
    #
    # SAE, AAE and ChE are matched within that pair.
    # --------------------------------------------------------

    index_cols = [
        scenario_col,
        "arm",
    ]

    # Check that every scenario/arm has all 3 dialects
    completeness = (
        df.groupby(index_cols)["dialect"]
        .nunique()
    )

    incomplete = completeness[
        completeness != 3
    ]

    print(
        f"\nIncomplete scenario-arm pairs: "
        f"{len(incomplete)}"
    )

    if len(incomplete):
        validation_lines.append(
            f"{model_name}: WARNING - "
            f"{len(incomplete)} incomplete scenario-arm pairs"
        )
    else:
        validation_lines.append(
            f"{model_name}: PASS - all scenario-arm "
            f"pairs contain SAE, AAE and ChE"
        )

    # --------------------------------------------------------
    # Calculate effects
    # --------------------------------------------------------

    for outcome in OUTCOME_COLUMNS:

        pivot = df.pivot_table(
            index=index_cols,
            columns="dialect",
            values=outcome,
            aggfunc="first"
        )

        # Need SAE + AAE
        if not {"SAE", "AAE"}.issubset(pivot.columns):
            continue

        aae_difference = (
            pivot["AAE"] -
            pivot["SAE"]
        )

        result = paired_effect(
            aae_difference
        )

        result.update({
            "model": model_name,
            "dialect": "AAE",
            "comparison": "AAE - SAE",
            "outcome": outcome,
        })

        all_effects.append(result)

        # Need SAE + ChE
        if not {"SAE", "ChE"}.issubset(pivot.columns):
            continue

        che_difference = (
            pivot["ChE"] -
            pivot["SAE"]
        )

        result = paired_effect(
            che_difference
        )

        result.update({
            "model": model_name,
            "dialect": "ChE",
            "comparison": "ChE - SAE",
            "outcome": outcome,
        })

        all_effects.append(result)


# ============================================================
# COMBINE RESULTS
# ============================================================

effects = pd.DataFrame(all_effects)

effects = effects[
    [
        "model",
        "dialect",
        "comparison",
        "outcome",
        "estimate",
        "standard_error",
        "ci_low",
        "ci_high",
        "t_stat",
        "p_value",
        "n",
    ]
]

# ------------------------------------------------------------
# Multiple-comparison correction
# ------------------------------------------------------------
#
# 4 models × 2 dialect contrasts × 7 outcomes = 56 tests
#
# This is the full exploratory family for this analysis.
# ------------------------------------------------------------

effects["p_bonferroni"] = bonferroni(
    effects["p_value"].values
)

effects["significant_bonferroni"] = (
    effects["p_bonferroni"] < 0.05
)


# ============================================================
# SAVE TABLE
# ============================================================

effects_file = (
    OUTPUT_DIR /
    "paired_effects_all.csv"
)

effects.to_csv(
    effects_file,
    index=False
)

print("\n" + "=" * 80)
print("PAIRED EFFECT RESULTS")
print("=" * 80)

print(
    effects[
        [
            "model",
            "dialect",
            "outcome",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "p_bonferroni",
            "n",
        ]
    ].to_string(index=False)
)

print(
    f"\nSaved:\n{effects_file}"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

summary = (
    effects
    .groupby(
        ["model", "dialect"],
        as_index=False
    )
    .agg(
        mean_effect=("estimate", "mean"),
        min_effect=("estimate", "min"),
        max_effect=("estimate", "max"),
        significant_outcomes=(
            "significant_bonferroni",
            "sum"
        ),
        outcomes_tested=("outcome", "count"),
    )
)

summary_file = (
    OUTPUT_DIR /
    "model_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    f"Saved:\n{summary_file}"
)


# ============================================================
# VALIDATION REPORT
# ============================================================

validation_file = (
    OUTPUT_DIR /
    "validation_report.txt"
)

with open(validation_file, "w") as f:

    f.write(
        "DIALECT BIAS ANALYSIS VALIDATION REPORT\n"
    )

    f.write("=" * 80 + "\n\n")

    for line in validation_lines:
        f.write(line + "\n")

    f.write("\n")
    f.write(
        f"Total effect tests: {len(effects)}\n"
    )

    f.write(
        "Bonferroni family: "
        "4 models × 2 dialect contrasts × "
        "7 outcomes = 56 tests\n"
    )

    f.write(
        "Bonferroni alpha: "
        f"{0.05 / 56:.6f}\n"
    )

print(
    f"Saved:\n{validation_file}"
)


# ============================================================
# FOREST PLOT FUNCTION
# ============================================================

def make_forest_plot(
    data,
    outcome,
    output_file,
    title=None,
):

    plot_data = data[
        data["outcome"] == outcome
    ].copy()

    if plot_data.empty:
        print(
            f"No data available for {outcome}"
        )
        return

    # Order:
    # AAE then ChE within each model
    model_order = list(
        RESULT_FILES.keys()
    )

    rows = []

    for model in model_order:

        for dialect in ["AAE", "ChE"]:

            row = plot_data[
                (plot_data["model"] == model) &
                (plot_data["dialect"] == dialect)
            ]

            if len(row) == 0:
                continue

            rows.append(
                row.iloc[0]
            )

    plot_data = pd.DataFrame(rows)

    # Reverse for matplotlib vertical ordering
    plot_data = plot_data.iloc[::-1]

    y = np.arange(len(plot_data))

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    estimates = plot_data["estimate"].values

    lower = (
        estimates -
        plot_data["ci_low"].values
    )

    upper = (
        plot_data["ci_high"].values -
        estimates
    )

    ax.errorbar(
        estimates,
        y,
        xerr=[lower, upper],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=6,
    )

    # Zero line
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    labels = []

    for _, row in plot_data.iterrows():

        labels.append(
            f"{row['model']} — "
            f"{row['dialect']}"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel(
        "Paired mean difference "
        "(dialect − SAE)"
    )

    if title is None:
        title = (
            outcome.replace(
                "_", " "
            ).title()
            + ": Dialect Effect"
        )

    ax.set_title(title)

    ax.grid(
        axis="x",
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved forest plot:\n{output_file}"
    )


# ============================================================
# INDIVIDUAL FOREST PLOTS
# ============================================================

make_forest_plot(
    effects,
    "moral_acceptability",
    OUTPUT_DIR /
    "forest_plot_moral_acceptability.png",
)

make_forest_plot(
    effects,
    "recommendation",
    OUTPUT_DIR /
    "forest_plot_recommendation.png",
)


# ============================================================
# ALL OUTCOMES FOREST PLOT
# ============================================================

# Arrange every outcome vertically.
plot_data = effects.copy()

outcome_order = OUTCOME_COLUMNS

model_order = list(
    RESULT_FILES.keys()
)

plot_data["outcome_order"] = (
    plot_data["outcome"]
    .map({
        outcome: i
        for i, outcome in enumerate(
            outcome_order
        )
    })
)

plot_data["model_order"] = (
    plot_data["model"]
    .map({
        model: i
        for i, model in enumerate(
            model_order
        )
    })
)

plot_data["dialect_order"] = (
    plot_data["dialect"]
    .map({
        "AAE": 0,
        "ChE": 1,
    })
)

plot_data = plot_data.sort_values(
    [
        "outcome_order",
        "model_order",
        "dialect_order",
    ]
)

plot_data = plot_data.iloc[::-1]

y = np.arange(
    len(plot_data)
)

fig, ax = plt.subplots(
    figsize=(12, 18)
)

estimates = (
    plot_data["estimate"].values
)

lower = (
    estimates -
    plot_data["ci_low"].values
)

upper = (
    plot_data["ci_high"].values -
    estimates
)

ax.errorbar(
    estimates,
    y,
    xerr=[lower, upper],
    fmt="o",
    capsize=3,
    linewidth=1.2,
    markersize=5,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

labels = []

for _, row in plot_data.iterrows():

    outcome_label = (
        row["outcome"]
        .replace("_", " ")
        .title()
    )

    labels.append(
        f"{outcome_label} | "
        f"{row['model']} | "
        f"{row['dialect']}"
    )

ax.set_yticks(y)
ax.set_yticklabels(
    labels,
    fontsize=8,
)

ax.set_xlabel(
    "Paired mean difference "
    "(dialect − SAE)"
)

ax.set_title(
    "Dialect Effects Across Models and Judgment Dimensions"
)

ax.grid(
    axis="x",
    alpha=0.2
)

plt.tight_layout()

all_plot_file = (
    OUTPUT_DIR /
    "forest_plot_all_outcomes.png"
)

plt.savefig(
    all_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved:\n{all_plot_file}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"""
Output directory:

    {OUTPUT_DIR}/

Files:

    paired_effects_all.csv
    model_summary.csv
    validation_report.txt

    forest_plot_all_outcomes.png
    forest_plot_moral_acceptability.png
    forest_plot_recommendation.png
"""
)

print(
    "Negative estimates mean the AAE/ChE rating "
    "was LOWER than the corresponding SAE rating."
)

print("=" * 80)