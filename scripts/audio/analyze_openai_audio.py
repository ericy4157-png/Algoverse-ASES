import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "results/pilot/gpt-audio.csv"
OUTPUT_FOLDER = "results/analysis/gpt-audio"

os.makedirs(f"{OUTPUT_FOLDER}/plots", exist_ok=True)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows")

print("\nBenchmark counts:")
print(df["benchmark"].value_counts())


# ============================================================
# Metrics
# ============================================================

metrics = [
    "moral_acceptability",
    "responsibility",
    "trustworthiness",
    "compassion",
    "fairness",
    "consequences",
    "recommendation"
]


# ============================================================
# Statistical analysis
# ============================================================

results = []


for benchmark in df["benchmark"].dropna().unique():

    print()
    print("=" * 60)
    print(f"BENCHMARK: {benchmark}")
    print("=" * 60)

    benchmark_df = df[
        df["benchmark"] == benchmark
    ].copy()

    # --------------------------------------------------------
    # Pair SAE and AAE within scenario + path
    # --------------------------------------------------------

    paired = benchmark_df.pivot_table(
        index=["scenario_id", "path"],
        columns="dialect",
        values=metrics,
        aggfunc="first"
    )

    for metric in metrics:

        # Make sure both dialects exist
        try:
            sae = paired[(metric, "SAE")]
            aae = paired[(metric, "AAE")]
        except KeyError:
            print(f"Skipping {metric}: missing SAE/AAE")
            continue

        # ----------------------------------------------------
        # Remove missing pairs
        # ----------------------------------------------------

        valid = pd.concat(
            [sae, aae],
            axis=1
        ).dropna()

        if len(valid) == 0:
            print(f"Skipping {metric}: no valid pairs")
            continue

        sae = valid.iloc[:, 0]
        aae = valid.iloc[:, 1]

        difference = aae - sae

        # ----------------------------------------------------
        # Wilcoxon
        # ----------------------------------------------------

        if len(valid) >= 2 and not np.all(difference == 0):

            try:
                stat, p = wilcoxon(
                    sae,
                    aae
                )
            except ValueError:
                stat, p = None, None

        else:
            stat, p = None, None

        # ----------------------------------------------------
        # Cohen's dz
        # ----------------------------------------------------

        if len(difference) >= 2:

            sd = difference.std(ddof=1)

            if sd != 0:
                dz = difference.mean() / sd
            else:
                dz = 0

        else:
            dz = None

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        mean_difference = difference.mean()

        if mean_difference > 0:
            direction = "AAE_higher"

        elif mean_difference < 0:
            direction = "AAE_lower"

        else:
            direction = "no_difference"

        # ----------------------------------------------------
        # Significance
        # ----------------------------------------------------

        if p is not None:
            significant = p < 0.05
        else:
            significant = False

        # ----------------------------------------------------
        # Effect size category
        # ----------------------------------------------------

        if dz is None:
            effect_category = "not_available"

        elif abs(dz) < 0.20:
            effect_category = "negligible"

        elif abs(dz) < 0.50:
            effect_category = "small"

        elif abs(dz) < 0.80:
            effect_category = "moderate"

        else:
            effect_category = "large"

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "model": INPUT_FILE.split("/")[-1]
                .replace(".csv", ""),

            "benchmark": benchmark,

            "metric": metric,

            "SAE_mean": round(sae.mean(), 3),

            "AAE_mean": round(aae.mean(), 3),

            "AAE_minus_SAE": round(
                mean_difference,
                3
            ),

            "wilcoxon_statistic": stat,

            "p_value": p,

            "cohens_dz": (
                round(dz, 3)
                if dz is not None
                else None
            ),

            "direction": direction,

            "significant_p05": significant,

            "effect_size_category": effect_category,

            "n": len(valid)

        })


# ============================================================
# Save statistics
# ============================================================

results_df = pd.DataFrame(results)

statistics_file = (
    f"{OUTPUT_FOLDER}/statistics.csv"
)

results_df.to_csv(
    statistics_file,
    index=False
)


print()
print("=" * 60)
print("STATISTICAL RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

print()
print(f"Saved statistics to:")
print(statistics_file)


# ============================================================
# Create plots
# ============================================================

for benchmark in df["benchmark"].dropna().unique():

    benchmark_df = df[
        df["benchmark"] == benchmark
    ].copy()

    safe_benchmark = (
        str(benchmark)
        .replace(" ", "_")
        .replace("/", "_")
    )

    benchmark_plot_folder = (
        f"{OUTPUT_FOLDER}/plots/{safe_benchmark}"
    )

    os.makedirs(
        benchmark_plot_folder,
        exist_ok=True
    )

    for metric in metrics:

        plot_df = benchmark_df[
            [
                "dialect",
                metric
            ]
        ].dropna()

        if len(plot_df) == 0:
            continue

        plt.figure(
            figsize=(6, 4)
        )

        plot_df.boxplot(
            column=metric,
            by="dialect"
        )

        plt.title(
            f"{benchmark}: {metric}"
        )

        plt.suptitle("")

        plt.ylabel(
            "Rating (1-7)"
        )

        plt.xlabel(
            "Dialect"
        )

        plt.ylim(
            1,
            7
        )

        plt.savefig(
            f"{benchmark_plot_folder}/{metric}.png",
            bbox_inches="tight"
        )

        plt.close()


print()
print("Finished analysis.")
print(
    f"Plots saved in {OUTPUT_FOLDER}/plots/"
)