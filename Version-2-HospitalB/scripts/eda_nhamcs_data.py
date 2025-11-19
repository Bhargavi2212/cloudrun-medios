"""
Comprehensive Exploratory Data Analysis (EDA) for NHAMCS Triage Dataset.

This script performs:
1. Univariate Analysis (numerical and categorical)
2. Bivariate Analysis (num-num, cat-cat, num-cat)
3. Multivariate Analysis
4. Generates visualizations and summary reports
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

DATA_DIR = Path(r"D:\Hackathons\Cloud Run\Version -2\data")
OUTPUT_DIR = Path(r"D:\Hackathons\Cloud Run\Version -2\data\eda_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the combined NHAMCS dataset."""
    df = pd.read_csv(DATA_DIR / "nhamcs_triage_dataset.csv")
    print(f"Loaded dataset: {len(df):,} records, {len(df.columns)} columns")
    return df


def univariate_analysis_numerical(df: pd.DataFrame) -> None:
    """Perform univariate analysis for numerical variables."""
    print("\n" + "=" * 70)
    print("UNIVARIATE ANALYSIS: NUMERICAL VARIABLES")
    print("=" * 70)

    numerical_vars = [
        "pulse",
        "respiration",
        "sbp",
        "dbp",
        "o2_sat",
        "temp_c",
        "gcs",
        "pain",
        "age",
        "wait_time",
        "length_of_visit",
        "past_visits",
    ]

    numerical_vars = [v for v in numerical_vars if v in df.columns]

    # Summary statistics
    print("\n1. SUMMARY STATISTICS")
    print("-" * 70)
    summary = df[numerical_vars].describe()
    print(summary.round(2))

    # Save summary to CSV
    summary.to_csv(OUTPUT_DIR / "univariate_numerical_summary.csv")

    # Missing values
    print("\n2. MISSING VALUES")
    print("-" * 70)
    missing = df[numerical_vars].isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    print(missing_df[missing_df["Missing Count"] > 0])

    # Distribution plots
    print("\n3. GENERATING DISTRIBUTION PLOTS...")
    n_vars = len(numerical_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, var in enumerate(numerical_vars):
        ax = axes[idx]
        data = df[var].dropna()

        # Histogram with KDE
        ax.hist(data, bins=50, alpha=0.7, edgecolor="black", density=True)
        ax.axvline(
            data.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {data.mean():.2f}",
        )
        ax.axvline(
            data.median(),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {data.median():.2f}",
        )
        ax.set_xlabel(var.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {var.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(numerical_vars), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "univariate_numerical_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"  Saved: {OUTPUT_DIR / 'univariate_numerical_distributions.png'}")
    plt.close()

    # Box plots
    print("\n4. GENERATING BOX PLOTS...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, var in enumerate(numerical_vars):
        ax = axes[idx]
        data = df[var].dropna()

        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        ax.set_ylabel(var.replace("_", " ").title())
        ax.set_title(f"Box Plot: {var.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(len(numerical_vars), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "univariate_numerical_boxplots.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'univariate_numerical_boxplots.png'}")
    plt.close()

    # Skewness and Kurtosis
    print("\n5. SKEWNESS AND KURTOSIS")
    print("-" * 70)
    skew_kurt = pd.DataFrame(
        {
            "Skewness": [stats.skew(df[var].dropna()) for var in numerical_vars],
            "Kurtosis": [stats.kurtosis(df[var].dropna()) for var in numerical_vars],
        },
        index=numerical_vars,
    )
    print(skew_kurt.round(3))
    skew_kurt.to_csv(OUTPUT_DIR / "univariate_numerical_skew_kurt.csv")


def outlier_analysis(df: pd.DataFrame) -> None:
    """Perform detailed outlier analysis using box plots and IQR method."""
    print("\n" + "=" * 70)
    print("OUTLIER ANALYSIS")
    print("=" * 70)

    numerical_vars = [
        "pulse",
        "respiration",
        "sbp",
        "dbp",
        "o2_sat",
        "temp_c",
        "gcs",
        "pain",
        "age",
        "wait_time",
        "length_of_visit",
    ]
    numerical_vars = [v for v in numerical_vars if v in df.columns]

    # Outlier detection using IQR method
    print("\n1. OUTLIER DETECTION (IQR METHOD)")
    print("-" * 70)

    outlier_stats = []

    for var in numerical_vars:
        data = df[var].dropna()

        if len(data) == 0:
            continue

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = outlier_count / len(data) * 100

        outlier_stats.append(
            {
                "Variable": var,
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
                "Outlier Count": outlier_count,
                "Outlier %": outlier_pct,
                "Min Value": data.min(),
                "Max Value": data.max(),
                "Min Outlier": outliers.min() if outlier_count > 0 else None,
                "Max Outlier": outliers.max() if outlier_count > 0 else None,
            }
        )

        print(f"\n{var.replace('_', ' ').title()}:")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        print(f"  Outliers: {outlier_count:,} ({outlier_pct:.2f}%)")
        if outlier_count > 0:
            print(
                f"  Min Outlier: {outliers.min():.2f}, Max Outlier: {outliers.max():.2f}"
            )

    outlier_df = pd.DataFrame(outlier_stats)
    outlier_df.to_csv(OUTPUT_DIR / "outlier_statistics.csv", index=False)

    # Enhanced box plots with outlier highlighting
    print("\n2. GENERATING ENHANCED BOX PLOTS WITH OUTLIER HIGHLIGHTING...")
    n_vars = len(numerical_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, var in enumerate(numerical_vars):
        ax = axes[idx]
        data = df[var].dropna()

        if len(data) == 0:
            axes[idx].axis("off")
            continue

        # Create box plot
        ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            meanprops=dict(color="green", linewidth=2, linestyle="--"),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
        )

        # Calculate outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]

        # Highlight outliers
        if len(outliers) > 0:
            [i for i, val in enumerate(data) if val in outliers.values]
            ax.scatter(
                [1] * len(outliers),
                outliers.values,
                color="red",
                alpha=0.5,
                s=30,
                zorder=3,
                label=f"Outliers ({len(outliers)})",
            )

        ax.set_ylabel(var.replace("_", " ").title(), fontsize=11)
        ax.set_title(
            f"Box Plot: {var.replace('_', ' ').title()}\n"
            f"Outliers: {len(outliers):,} ({len(outliers)/len(data)*100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add IQR bounds as reference lines
        ax.axhline(
            lower_bound,
            color="orange",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="IQR Bounds",
        )
        ax.axhline(upper_bound, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.legend(loc="upper right", fontsize=9)

    for idx in range(len(numerical_vars), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "outlier_boxplots_detailed.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'outlier_boxplots_detailed.png'}")
    plt.close()

    # Outlier distribution by ESI level
    print("\n3. OUTLIER DISTRIBUTION BY ESI LEVEL...")
    if "esi_level" in df.columns:
        n_vars = len(numerical_vars)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7 * n_rows))
        axes = axes.flatten() if n_vars > 1 else [axes]

        for idx, var in enumerate(numerical_vars):
            ax = axes[idx]
            data = df[[var, "esi_level"]].dropna()

            if len(data) == 0:
                axes[idx].axis("off")
                continue

            # Calculate IQR bounds for entire dataset
            Q1 = data[var].quantile(0.25)
            Q3 = data[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Group by ESI level
            esi_levels = sorted(data["esi_level"].unique())
            data_to_plot = []
            labels = []
            outlier_counts = []

            for esi in esi_levels:
                esi_data = data[data["esi_level"] == esi][var]
                data_to_plot.append(esi_data)
                labels.append(f"ESI {int(esi)}")

                # Count outliers for this ESI level
                outliers = esi_data[(esi_data < lower_bound) | (esi_data > upper_bound)]
                outlier_counts.append(len(outliers))

            # Create box plot
            ax.boxplot(
                data_to_plot,
                labels=labels,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                boxprops=dict(facecolor="lightblue", alpha=0.7),
                medianprops=dict(color="red", linewidth=2),
                meanprops=dict(color="green", linewidth=2, linestyle="--"),
            )

            # Add outlier counts to title
            outlier_info = ", ".join(
                [
                    f"ESI {int(esi)}: {count}"
                    for esi, count in zip(esi_levels, outlier_counts, strict=False)
                ]
            )

            ax.set_ylabel(var.replace("_", " ").title(), fontsize=11)
            ax.set_xlabel("ESI Level", fontsize=11)
            ax.set_title(
                f"{var.replace('_', ' ').title()} by ESI Level\n"
                f"Outliers: {outlier_info}",
                fontsize=11,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis="x", rotation=45)

        for idx in range(len(numerical_vars), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "outlier_boxplots_by_esi.png", dpi=300, bbox_inches="tight"
        )
        print(f"  Saved: {OUTPUT_DIR / 'outlier_boxplots_by_esi.png'}")
        plt.close()

    # Outlier summary table
    print("\n4. OUTLIER SUMMARY")
    print("-" * 70)
    summary_table = outlier_df[
        ["Variable", "Outlier Count", "Outlier %", "Min Outlier", "Max Outlier"]
    ].copy()
    summary_table = summary_table.sort_values("Outlier %", ascending=False)
    print(summary_table.to_string(index=False))
    summary_table.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)

    # Create visualization of outlier percentages
    print("\n5. GENERATING OUTLIER PERCENTAGE VISUALIZATION...")
    plt.figure(figsize=(12, 8))
    sorted_df = outlier_df.sort_values("Outlier %", ascending=True)
    colors = [
        "red" if pct > 10 else "orange" if pct > 5 else "green"
        for pct in sorted_df["Outlier %"]
    ]

    bars = plt.barh(
        range(len(sorted_df)), sorted_df["Outlier %"], color=colors, alpha=0.7
    )
    plt.yticks(
        range(len(sorted_df)), sorted_df["Variable"].str.replace("_", " ").str.title()
    )
    plt.xlabel("Outlier Percentage (%)", fontsize=12)
    plt.title(
        "Outlier Percentage by Variable\n(Red: >10%, Orange: 5-10%, Green: <5%)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for _i, (bar, pct) in enumerate(zip(bars, sorted_df["Outlier %"], strict=False)):
        plt.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "outlier_percentage_chart.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'outlier_percentage_chart.png'}")
    plt.close()


def univariate_analysis_categorical(df: pd.DataFrame) -> None:
    """Perform univariate analysis for categorical variables."""
    print("\n" + "=" * 70)
    print("UNIVARIATE ANALYSIS: CATEGORICAL VARIABLES")
    print("=" * 70)

    categorical_vars = [
        "esi_level",
        "sex",
        "month",
        "day_of_week",
        "ambulance_arrival",
        "injury",
        "cebvd",
        "chf",
        "ed_dialysis",
        "hiv",
        "diabetes",
        "no_chronic_conditions",
        "seen_72h",
        "discharged_7d",
    ]

    categorical_vars = [v for v in categorical_vars if v in df.columns]

    # Frequency tables
    print("\n1. FREQUENCY TABLES")
    print("-" * 70)
    freq_tables = {}

    for var in categorical_vars:
        freq = df[var].value_counts().sort_index()
        freq_pct = (df[var].value_counts(normalize=True) * 100).sort_index()
        freq_table = pd.DataFrame({"Count": freq, "Percentage": freq_pct.round(2)})
        freq_tables[var] = freq_table
        print(f"\n{var.replace('_', ' ').title()}:")
        print(freq_table.head(10))

    # Save frequency tables
    with pd.ExcelWriter(
        OUTPUT_DIR / "univariate_categorical_frequencies.xlsx"
    ) as writer:
        for var, table in freq_tables.items():
            table.to_excel(writer, sheet_name=var[:31])  # Excel sheet name limit

    # Bar plots
    print("\n2. GENERATING BAR PLOTS...")
    n_vars = len(categorical_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, var in enumerate(categorical_vars):
        ax = axes[idx]
        counts = df[var].value_counts().sort_index()

        bars = ax.bar(
            range(len(counts)),
            counts.values,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Frequency: {var.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for idx in range(len(categorical_vars), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "univariate_categorical_barplots.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'univariate_categorical_barplots.png'}")
    plt.close()

    # ESI Level distribution (target variable - special attention)
    print("\n3. ESI LEVEL DISTRIBUTION (TARGET VARIABLE)")
    print("-" * 70)
    esi_dist = df["esi_level"].value_counts().sort_index()
    esi_pct = (df["esi_level"].value_counts(normalize=True) * 100).sort_index()
    esi_table = pd.DataFrame({"Count": esi_dist, "Percentage": esi_pct.round(2)})
    print(esi_table)

    # ESI Level pie chart
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("Set2", len(esi_dist))
    plt.pie(
        esi_dist.values,
        labels=[f"ESI {int(x)}" for x in esi_dist.index],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    plt.title(
        "ESI Level Distribution (Target Variable)", fontsize=14, fontweight="bold"
    )
    plt.savefig(OUTPUT_DIR / "esi_level_distribution.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'esi_level_distribution.png'}")
    plt.close()


def bivariate_analysis_num_num(df: pd.DataFrame) -> None:
    """Perform bivariate analysis for numerical-numerical pairs."""
    print("\n" + "=" * 70)
    print("BIVARIATE ANALYSIS: NUMERICAL-NUMERICAL")
    print("=" * 70)

    numerical_vars = [
        "pulse",
        "respiration",
        "sbp",
        "dbp",
        "o2_sat",
        "temp_c",
        "gcs",
        "pain",
        "age",
        "wait_time",
        "length_of_visit",
    ]
    numerical_vars = [v for v in numerical_vars if v in df.columns]

    # Correlation matrix
    print("\n1. CORRELATION MATRIX")
    print("-" * 70)
    corr_matrix = df[numerical_vars].corr()
    print(corr_matrix.round(3))
    corr_matrix.to_csv(OUTPUT_DIR / "bivariate_numerical_correlation.csv")

    # Correlation heatmap
    print("\n2. GENERATING CORRELATION HEATMAP...")
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Correlation Heatmap: Numerical Variables", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "bivariate_numerical_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"  Saved: {OUTPUT_DIR / 'bivariate_numerical_correlation_heatmap.png'}")
    plt.close()

    # Scatter plots for top correlations
    print("\n3. TOP CORRELATIONS")
    print("-" * 70)
    # Get upper triangle of correlation matrix
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append(
                (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            )

    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    top_corr = corr_pairs[:6]  # Top 6 correlations

    print("Top correlations:")
    for var1, var2, corr in top_corr:
        print(f"  {var1} - {var2}: {corr:.3f}")

    # Scatter plots for top correlations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (var1, var2, corr) in enumerate(top_corr):
        ax = axes[idx]
        sample_size = min(5000, len(df))  # Sample for performance
        sample_df = df[[var1, var2]].dropna().sample(n=sample_size, random_state=42)

        ax.scatter(sample_df[var1], sample_df[var2], alpha=0.3, s=10)
        ax.set_xlabel(var1.replace("_", " ").title())
        ax.set_ylabel(var2.replace("_", " ").title())
        ax.set_title(f"{var1} vs {var2}\n(r={corr:.3f})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "bivariate_numerical_scatterplots.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"  Saved: {OUTPUT_DIR / 'bivariate_numerical_scatterplots.png'}")
    plt.close()


def bivariate_analysis_cat_cat(df: pd.DataFrame) -> None:
    """Perform bivariate analysis for categorical-categorical pairs."""
    print("\n" + "=" * 70)
    print("BIVARIATE ANALYSIS: CATEGORICAL-CATEGORICAL")
    print("=" * 70)

    categorical_vars = [
        "esi_level",
        "sex",
        "ambulance_arrival",
        "injury",
        "cebvd",
        "chf",
        "ed_dialysis",
        "hiv",
        "diabetes",
        "no_chronic_conditions",
    ]
    categorical_vars = [v for v in categorical_vars if v in df.columns]

    # Cross-tabulations with ESI level (target)
    print("\n1. CROSS-TABULATIONS WITH ESI LEVEL (TARGET)")
    print("-" * 70)

    crosstabs = {}
    for var in categorical_vars:
        if var == "esi_level":
            continue

        crosstab = pd.crosstab(df[var], df["esi_level"], margins=True)
        crosstab_pct = (
            pd.crosstab(df[var], df["esi_level"], normalize="index", margins=True) * 100
        )

        crosstabs[var] = {"count": crosstab, "percentage": crosstab_pct}

        print(f"\n{var.replace('_', ' ').title()} vs ESI Level (Counts):")
        print(crosstab)
        print(f"\n{var.replace('_', ' ').title()} vs ESI Level (Row %):")
        print(crosstab_pct.round(2))

    # Save crosstabs
    with pd.ExcelWriter(OUTPUT_DIR / "bivariate_categorical_crosstabs.xlsx") as writer:
        for var, tables in crosstabs.items():
            tables["count"].to_excel(writer, sheet_name=f"{var[:25]}_count")
            tables["percentage"].to_excel(writer, sheet_name=f"{var[:25]}_pct")

    # Chi-square tests
    print("\n2. CHI-SQUARE TESTS")
    print("-" * 70)
    chi2_results = []

    for var in categorical_vars:
        if var == "esi_level":
            continue

        contingency = pd.crosstab(df[var], df["esi_level"])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        chi2_results.append(
            {
                "Variable": var,
                "Chi-square": chi2,
                "p-value": p_value,
                "Degrees of Freedom": dof,
                "Significant": "Yes" if p_value < 0.05 else "No",
            }
        )

        print(
            f"{var}: Chi2={chi2:.2f}, p={p_value:.4f}, significant={'Yes' if p_value < 0.05 else 'No'}"
        )

    chi2_df = pd.DataFrame(chi2_results)
    chi2_df.to_csv(OUTPUT_DIR / "bivariate_categorical_chi2_tests.csv", index=False)

    # Stacked bar charts
    print("\n3. GENERATING STACKED BAR CHARTS...")
    n_vars = len([v for v in categorical_vars if v != "esi_level"])
    n_cols = 2
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    idx = 0
    for var in categorical_vars:
        if var == "esi_level":
            continue

        ax = axes[idx]
        crosstab = pd.crosstab(df[var], df["esi_level"], normalize="index") * 100

        crosstab.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=sns.color_palette("Set2", len(crosstab.columns)),
        )
        ax.set_xlabel(var.replace("_", " ").title())
        ax.set_ylabel("Percentage")
        ax.set_title(f"ESI Level Distribution by {var.replace('_', ' ').title()}")
        ax.legend(title="ESI Level", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

        idx += 1

    for idx in range(len([v for v in categorical_vars if v != "esi_level"]), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "bivariate_categorical_stacked_bars.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"  Saved: {OUTPUT_DIR / 'bivariate_categorical_stacked_bars.png'}")
    plt.close()


def bivariate_analysis_num_cat(df: pd.DataFrame) -> None:
    """Perform bivariate analysis for numerical-categorical pairs."""
    print("\n" + "=" * 70)
    print("BIVARIATE ANALYSIS: NUMERICAL-CATEGORICAL")
    print("=" * 70)

    numerical_vars = [
        "pulse",
        "respiration",
        "sbp",
        "dbp",
        "o2_sat",
        "temp_c",
        "gcs",
        "pain",
        "age",
    ]
    numerical_vars = [v for v in numerical_vars if v in df.columns]

    categorical_vars = ["esi_level", "sex", "ambulance_arrival", "injury"]
    categorical_vars = [v for v in categorical_vars if v in df.columns]

    # Group statistics by categorical variable
    print("\n1. GROUP STATISTICS BY CATEGORICAL VARIABLES")
    print("-" * 70)

    group_stats = {}
    for cat_var in categorical_vars:
        print(f"\n{cat_var.replace('_', ' ').title()}:")
        for num_var in numerical_vars:
            grouped = df.groupby(cat_var)[num_var].agg(
                ["mean", "median", "std", "count"]
            )
            group_stats[f"{cat_var}_{num_var}"] = grouped
            print(f"\n  {num_var.replace('_', ' ').title()}:")
            print(grouped.round(2))

    # Save group statistics
    with pd.ExcelWriter(OUTPUT_DIR / "bivariate_num_cat_group_stats.xlsx") as writer:
        for key, stats_df in group_stats.items():
            stats_df.to_excel(writer, sheet_name=key[:31])

    # Box plots: numerical by categorical
    print("\n2. GENERATING BOX PLOTS (NUMERICAL BY CATEGORICAL)...")

    # Focus on ESI level (target) vs numerical variables
    n_vars = len(numerical_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, num_var in enumerate(numerical_vars):
        ax = axes[idx]
        data_to_plot = [
            df[df["esi_level"] == esi][num_var].dropna()
            for esi in sorted(df["esi_level"].unique())
        ]

        bp = ax.boxplot(
            data_to_plot,
            labels=[f"ESI {int(esi)}" for esi in sorted(df["esi_level"].unique())],
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax.set_xlabel("ESI Level")
        ax.set_ylabel(num_var.replace("_", " ").title())
        ax.set_title(f"{num_var.replace('_', ' ').title()} by ESI Level")
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(len(numerical_vars), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "bivariate_num_cat_boxplots_esi.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'bivariate_num_cat_boxplots_esi.png'}")
    plt.close()

    # ANOVA tests
    print("\n3. ANOVA TESTS (NUMERICAL BY ESI LEVEL)")
    print("-" * 70)
    anova_results = []

    for num_var in numerical_vars:
        groups = [
            df[df["esi_level"] == esi][num_var].dropna()
            for esi in sorted(df["esi_level"].unique())
        ]

        f_stat, p_value = stats.f_oneway(*groups)

        anova_results.append(
            {
                "Variable": num_var,
                "F-statistic": f_stat,
                "p-value": p_value,
                "Significant": "Yes" if p_value < 0.05 else "No",
            }
        )

        print(
            f"{num_var}: F={f_stat:.2f}, p={p_value:.4f}, significant={'Yes' if p_value < 0.05 else 'No'}"
        )

    anova_df = pd.DataFrame(anova_results)
    anova_df.to_csv(OUTPUT_DIR / "bivariate_num_cat_anova_tests.csv", index=False)


def multivariate_analysis(df: pd.DataFrame) -> None:
    """Perform multivariate analysis."""
    print("\n" + "=" * 70)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 70)

    # Feature importance correlation with target
    print("\n1. CORRELATION WITH TARGET (ESI LEVEL)")
    print("-" * 70)

    numerical_vars = [
        "pulse",
        "respiration",
        "sbp",
        "dbp",
        "o2_sat",
        "temp_c",
        "gcs",
        "pain",
        "age",
    ]
    numerical_vars = [v for v in numerical_vars if v in df.columns]

    target_corr = (
        df[[*numerical_vars, "esi_level"]]
        .corr()["esi_level"]
        .sort_values(ascending=False)
    )
    target_corr = target_corr.drop("esi_level")

    print(target_corr.round(3))
    target_corr.to_csv(OUTPUT_DIR / "multivariate_target_correlation.csv")

    # Correlation with target visualization
    plt.figure(figsize=(10, 6))
    target_corr.plot(kind="barh", color="steelblue")
    plt.xlabel("Correlation with ESI Level")
    plt.title(
        "Feature Correlation with Target (ESI Level)", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multivariate_target_correlation.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'multivariate_target_correlation.png'}")
    plt.close()

    # Pair plot for key variables
    print("\n2. GENERATING PAIR PLOT (KEY VARIABLES)...")
    key_vars = ["esi_level", "pulse", "sbp", "o2_sat", "temp_c", "age"]
    key_vars = [v for v in key_vars if v in df.columns]

    # Sample for performance
    sample_df = df[key_vars].dropna().sample(n=min(5000, len(df)), random_state=42)

    pair_plot = sns.pairplot(
        sample_df, hue="esi_level", palette="Set2", diag_kind="kde"
    )
    pair_plot.fig.suptitle(
        "Pair Plot: Key Variables by ESI Level", y=1.02, fontsize=14, fontweight="bold"
    )
    plt.savefig(OUTPUT_DIR / "multivariate_pairplot.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'multivariate_pairplot.png'}")
    plt.close()

    # Missing value patterns
    print("\n3. MISSING VALUE PATTERNS")
    print("-" * 70)
    missing_pattern = df.isnull().sum().sort_values(ascending=False)
    missing_pattern = missing_pattern[missing_pattern > 0]

    if len(missing_pattern) > 0:
        print(missing_pattern)
        plt.figure(figsize=(12, 6))
        missing_pattern.plot(kind="barh", color="coral")
        plt.xlabel("Missing Count")
        plt.title("Missing Value Patterns", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "multivariate_missing_patterns.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  Saved: {OUTPUT_DIR / 'multivariate_missing_patterns.png'}")
        plt.close()
    else:
        print("  No missing values found.")


def generate_summary_report(df: pd.DataFrame) -> None:
    """Generate a summary report."""
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)

    report = []
    report.append("=" * 70)
    report.append("NHAMCS TRIAGE DATASET - EXPLORATORY DATA ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("Dataset Overview:")
    report.append(f"  - Total Records: {len(df):,}")
    report.append(f"  - Total Features: {len(df.columns)}")
    report.append("  - Target Variable: esi_level")
    report.append("")

    report.append("Target Variable (ESI Level) Distribution:")
    esi_dist = df["esi_level"].value_counts().sort_index()
    for level, count in esi_dist.items():
        pct = count / len(df) * 100
        report.append(f"  - ESI {int(level)}: {count:,} ({pct:.1f}%)")
    report.append("")

    report.append("Key Findings:")
    report.append(
        "  1. Univariate analysis: See distribution plots and summary statistics"
    )
    report.append(
        "  2. Outlier analysis: See box plots with IQR-based outlier detection"
    )
    report.append(
        "  3. Bivariate analysis: See correlation matrices and cross-tabulations"
    )
    report.append("  4. Multivariate analysis: See pair plots and feature correlations")
    report.append("")
    report.append("Output Files:")
    report.append("  - All visualizations saved to: data/eda_output/")
    report.append("  - All summary tables saved as CSV/Excel files")
    report.append("=" * 70)

    report_text = "\n".join(report)
    print(report_text)

    with open(OUTPUT_DIR / "eda_summary_report.txt", "w") as f:
        f.write(report_text)

    print(f"\n  Saved: {OUTPUT_DIR / 'eda_summary_report.txt'}")


def main() -> None:
    """Main EDA pipeline."""
    print("=" * 70)
    print("NHAMCS TRIAGE DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Load data
    df = load_data()

    # Univariate Analysis
    univariate_analysis_numerical(df)
    univariate_analysis_categorical(df)

    # Outlier Analysis
    outlier_analysis(df)

    # Bivariate Analysis
    bivariate_analysis_num_num(df)
    bivariate_analysis_cat_cat(df)
    bivariate_analysis_num_cat(df)

    # Multivariate Analysis
    multivariate_analysis(df)

    # Generate summary report
    generate_summary_report(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
