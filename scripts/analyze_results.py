import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os


# ----------------------------
# Configuration
# ----------------------------

INPUT_FILE = "results/pilot/gpt5_v2.csv"

OUTPUT_FOLDER = "results/analysis/gpt5_v2"

os.makedirs(
    f"{OUTPUT_FOLDER}/plots",
    exist_ok=True
)


# ----------------------------
# Load data
# ----------------------------

df = pd.read_csv(INPUT_FILE)


# ----------------------------
# Metrics
# ----------------------------

metrics = [
    "moral_acceptability",
    "responsibility",
    "trustworthiness",
    "compassion",
    "fairness",
    "consequences",
    "recommendation"
]


# ----------------------------
# Convert SAE/AAE into pairs
# ----------------------------

paired = df.pivot_table(
    index=[
        "scenario_id",
        "path"
    ],
    columns="dialect",
    values=metrics
)


results = []


# ----------------------------
# Statistical testing
# ----------------------------

for metric in metrics:

    sae = paired[(metric, "SAE")]
    aae = paired[(metric, "AAE")]


    # Remove missing values
    valid = pd.concat(
        [sae, aae],
        axis=1
    ).dropna()


    sae = valid.iloc[:, 0]
    aae = valid.iloc[:, 1]


    difference = aae - sae


    # Wilcoxon test
    if len(valid) > 0:
        stat, p = wilcoxon(
            sae,
            aae
        )
    else:
        stat, p = None, None


    # Cohen's dz
    if difference.std() != 0:
        dz = (
            difference.mean()
            /
            difference.std()
        )
    else:
        dz = 0


    results.append({

        "metric": metric,

        "SAE_mean": round(
            sae.mean(),
            3
        ),

        "AAE_mean": round(
            aae.mean(),
            3
        ),

        "AAE_minus_SAE": round(
            difference.mean(),
            3
        ),

        "wilcoxon_statistic": stat,

        "p_value": p,

        "cohens_dz": round(
            dz,
            3
        ),

        "n": len(valid)

    })


# ----------------------------
# Save statistics
# ----------------------------

results_df = pd.DataFrame(results)


results_df.to_csv(
    f"{OUTPUT_FOLDER}/statistics.csv",
    index=False
)


print("=" * 60)
print("STATISTICAL RESULTS")
print("=" * 60)

print(results_df)


# ----------------------------
# Create plots
# ----------------------------

for metric in metrics:

    plot_df = df[
        [
            "dialect",
            metric
        ]
    ]


    plt.figure(
        figsize=(6,4)
    )


    plot_df.boxplot(
        column=metric,
        by="dialect"
    )


    plt.title(
        f"{metric}: SAE vs AAE"
    )

    plt.suptitle("")

    plt.ylabel(
        "Rating (1-7)"
    )


    plt.savefig(
        f"{OUTPUT_FOLDER}/plots/{metric}.png",
        bbox_inches="tight"
    )


    plt.close()


print()
print("Finished analysis.")
print(f"Files saved in {OUTPUT_FOLDER}/")