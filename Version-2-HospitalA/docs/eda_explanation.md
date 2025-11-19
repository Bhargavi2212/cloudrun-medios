# Detailed Explanation of Exploratory Data Analysis (EDA)

## Overview

We performed a comprehensive Exploratory Data Analysis (EDA) on the NHAMCS Triage Dataset containing **163,736 patient records** with **23 features**. The goal was to understand the data structure, relationships, and patterns before building the triage prediction model.

---

## 1. UNIVARIATE ANALYSIS

Univariate analysis examines each variable **independently** to understand its distribution, central tendencies, and characteristics.

### 1.1 Numerical Variables Analysis

**Variables Analyzed:**
- Vital Signs: `pulse`, `respiration`, `sbp` (systolic BP), `dbp` (diastolic BP), `o2_sat`, `temp_c`, `gcs`
- Clinical: `pain`, `age`
- Operational: `wait_time`, `length_of_visit`, `past_visits`

#### A. Summary Statistics
- **What we calculated:**
  - **Count**: Number of non-null values
  - **Mean**: Average value
  - **Median**: Middle value (50th percentile)
  - **Standard Deviation**: Measure of spread/variability
  - **Min/Max**: Range of values
  - **Quartiles (25%, 50%, 75%)**: Distribution percentiles

- **Why it matters:** 
  - Identifies central tendencies (mean vs median)
  - Detects skewness (if mean ≠ median)
  - Understands data ranges and variability
  - Example: If mean pulse is 91 but median is 88, data is right-skewed (some very high pulse values)

#### B. Missing Value Analysis
- **What we did:** Counted and calculated percentage of missing values for each variable
- **Key Findings:**
  - GCS: 94.42% missing (most missing)
  - Past visits: 81.58% missing
  - O2 saturation: 23.93% missing
  - Pain: 22.86% missing
- **Why it matters:** Determines which variables need imputation or exclusion

#### C. Distribution Plots (Histograms with KDE)
- **What we created:** Histograms showing the frequency distribution of each numerical variable
- **Features:**
  - Density curves (KDE - Kernel Density Estimation) showing smooth distribution
  - Mean line (red dashed)
  - Median line (green dashed)
- **Why it matters:**
  - Identifies if data is normally distributed, skewed, or has multiple peaks
  - Helps decide if transformations are needed (e.g., log transform for skewed data)
  - Example: If pulse is normally distributed, we can use parametric tests

#### D. Box Plots
- **What we created:** Box plots showing:
  - **Box**: Interquartile Range (IQR) - middle 50% of data
  - **Whiskers**: Extend to 1.5×IQR from quartiles
  - **Outliers**: Points beyond whiskers (shown as dots)
  - **Median**: Line inside box
- **Why it matters:**
  - Visualizes spread and skewness
  - Identifies outliers quickly
  - Compares distributions across variables

#### E. Skewness and Kurtosis
- **Skewness:** Measures asymmetry
  - Positive: Right tail longer (mean > median)
  - Negative: Left tail longer (mean < median)
  - Zero: Symmetric distribution
- **Kurtosis:** Measures tail heaviness
  - Positive: Heavy tails (more outliers)
  - Negative: Light tails (fewer outliers)
  - Zero: Normal distribution
- **Why it matters:** Determines if data transformation is needed for modeling

### 1.2 Categorical Variables Analysis

**Variables Analyzed:**
- Target: `esi_level` (1-5)
- Demographics: `sex`, `month`, `day_of_week`
- Clinical: `ambulance_arrival`, `injury`, chronic conditions (`cebvd`, `chf`, `ed_dialysis`, `hiv`, `diabetes`, `no_chronic_conditions`)
- History: `seen_72h`, `discharged_7d`

#### A. Frequency Tables
- **What we created:** Count and percentage of each category
- **Example Output:**
  ```
  ESI Level:
  - ESI 1: 1,866 (1.1%)
  - ESI 2: 19,997 (12.2%)
  - ESI 3: 77,629 (47.4%) ← Most common
  - ESI 4: 54,519 (33.3%)
  - ESI 5: 9,725 (5.9%)
  ```
- **Why it matters:**
  - Identifies class imbalance (ESI 3 dominates at 47%)
  - Shows rare categories (ESI 1 only 1.1%)
  - Helps plan for imbalanced classification techniques

#### B. Bar Charts
- **What we created:** Bar plots showing frequency of each category
- **Features:** Value labels on top of bars
- **Why it matters:** Visual representation of category frequencies

#### C. ESI Level Distribution (Special Focus)
- **What we created:** Pie chart showing target variable distribution
- **Why it matters:** 
  - Confirms severe class imbalance
  - Critical for model evaluation (need stratified sampling, class weights)

---

## 2. OUTLIER ANALYSIS

### What is an Outlier?
An outlier is a data point that significantly differs from other observations. We used the **IQR (Interquartile Range) Method**:

**Method:**
1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
2. Calculate IQR = Q3 - Q1
3. Lower Bound = Q1 - 1.5 × IQR
4. Upper Bound = Q3 + 1.5 × IQR
5. Any value < Lower Bound or > Upper Bound is an outlier

### A. Outlier Detection Statistics
- **What we calculated for each variable:**
  - Q1, Q3, IQR
  - Lower and upper bounds
  - Number and percentage of outliers
  - Min and max outlier values

**Key Findings:**
- **Wait Time**: 9.50% outliers (92-1440 minutes) - extreme cases
- **Temperature**: 7.52% outliers (30.22-43.28°C) - hypothermia/hyperthermia
- **Length of Visit**: 6.44% outliers (527-5741 minutes) - very long stays
- **Respiration**: 5.72% outliers (1-59) - extreme respiratory rates
- **Pain & Age**: 0% outliers - well-behaved data

### B. Enhanced Box Plots with Outlier Highlighting
- **What we created:**
  - Box plots with outliers highlighted in **red**
  - IQR bounds shown as **orange dashed lines**
  - Mean (green) and median (red) lines
  - Outlier counts in titles
- **Why it matters:** Visual identification of outliers for each variable

### C. Outlier Distribution by ESI Level
- **What we created:** Box plots grouped by ESI level showing:
  - Distribution of each variable across acuity levels
  - Outlier counts per ESI level
- **Why it matters:**
  - Identifies if outliers are concentrated in specific acuity levels
  - Example: Are extreme vitals more common in ESI 1 (critical) patients?

### D. Outlier Percentage Visualization
- **What we created:** Horizontal bar chart color-coded by severity:
  - **Red**: >10% outliers (high concern)
  - **Orange**: 5-10% outliers (moderate concern)
  - **Green**: <5% outliers (low concern)
- **Why it matters:** Quick visual summary of which variables need outlier treatment

---

## 3. BIVARIATE ANALYSIS

Bivariate analysis examines **relationships between pairs of variables**.

### 3.1 Numerical-Numerical Analysis

#### A. Correlation Matrix
- **What we calculated:** Pearson correlation coefficients between all numerical variable pairs
- **Range:** -1 to +1
  - **+1**: Perfect positive correlation
  - **0**: No correlation
  - **-1**: Perfect negative correlation
- **Key Findings:**
  - **SBP ↔ DBP**: 0.616 (strong positive) - expected, both BP measures
  - **Pulse ↔ Respiration**: 0.526 (moderate positive) - both vital signs
  - **Age ↔ SBP**: 0.404 (moderate positive) - BP increases with age
  - **Age ↔ Pulse**: -0.401 (moderate negative) - pulse decreases with age
- **Why it matters:**
  - Identifies multicollinearity (highly correlated features)
  - Helps feature selection (remove redundant features)
  - Understands physiological relationships

#### B. Correlation Heatmap
- **What we created:** Color-coded matrix visualization
  - **Red**: Positive correlation
  - **Blue**: Negative correlation
  - **Intensity**: Strength of correlation
- **Why it matters:** Visual pattern recognition of relationships

#### C. Scatter Plots for Top Correlations
- **What we created:** Scatter plots for the 6 strongest correlations
- **Why it matters:**
  - Visualizes relationship patterns (linear, non-linear, clusters)
  - Identifies potential non-linear relationships
  - Detects data quality issues

### 3.2 Categorical-Categorical Analysis

#### A. Cross-Tabulations (Contingency Tables)
- **What we created:** Frequency tables showing combinations of categorical variables
- **Example:** ESI Level × Ambulance Arrival
  ```
            ESI 1  ESI 2  ESI 3  ESI 4  ESI 5
  Ambulance  500    2000   8000   5000   500
  Walk-in    1366   17997  69629  49519  9225
  ```
- **Why it matters:**
  - Identifies associations between categories
  - Example: Are ambulance arrivals more likely to be ESI 1-2?

#### B. Chi-Square Tests
- **What we calculated:** Statistical test for independence between categorical variables
- **Null Hypothesis:** Variables are independent (no relationship)
- **Result:** All tests were significant (p < 0.05)
- **Why it matters:**
  - Confirms statistical significance of relationships
  - Validates that categories are not independent
  - Example: ESI level is significantly associated with ambulance arrival

#### C. Stacked Bar Charts
- **What we created:** Bar charts showing ESI level distribution within each category
- **Why it matters:**
  - Visual comparison of ESI distribution across categories
  - Identifies which categories predict higher/lower acuity

### 3.3 Numerical-Categorical Analysis

#### A. Group Statistics
- **What we calculated:** Mean, median, std, count for numerical variables grouped by categorical variables
- **Example:** Pulse by ESI Level
  ```
  ESI Level  Mean Pulse  Median  Std
  ESI 1        105.2      102    25.3
  ESI 2         95.8       94    22.1
  ESI 3         89.5       88    21.5
  ESI 4         88.2       87    20.8
  ESI 5         87.1       86    19.9
  ```
- **Why it matters:**
  - Shows how numerical values differ across categories
  - Identifies which numerical features predict the target (ESI level)

#### B. Box Plots by Category
- **What we created:** Box plots of numerical variables grouped by ESI level
- **Why it matters:**
  - Visual comparison of distributions across categories
  - Identifies if distributions differ significantly
  - Example: Do ESI 1 patients have higher pulse rates?

#### C. ANOVA Tests
- **What we calculated:** Analysis of Variance - tests if means differ across groups
- **Null Hypothesis:** All group means are equal
- **Result:** All tests significant (p < 0.05) - means differ significantly
- **Why it matters:**
  - Confirms that numerical variables can distinguish between ESI levels
  - Validates feature importance for modeling
  - Example: Pulse significantly differs across ESI levels → good predictor

---

## 4. MULTIVARIATE ANALYSIS

Multivariate analysis examines **relationships among multiple variables simultaneously**.

### A. Correlation with Target (ESI Level)
- **What we calculated:** Correlation of each numerical feature with ESI level
- **Key Findings:**
  - **Age**: -0.239 (strongest negative) - older patients → lower acuity
  - **SBP**: -0.092 (moderate negative) - higher BP → lower acuity
  - **GCS**: +0.077 (strongest positive) - higher GCS → higher acuity
  - **O2 Sat**: +0.060 (positive) - higher O2 → higher acuity
- **Why it matters:**
  - Identifies most predictive features
  - Guides feature engineering
  - Helps prioritize features for modeling

### B. Pair Plot
- **What we created:** Grid of scatter plots showing relationships between key variables, colored by ESI level
- **Variables:** ESI level, pulse, SBP, O2 sat, temp_c, age
- **Why it matters:**
  - Visualizes complex multi-variable relationships
  - Identifies clusters or patterns by ESI level
  - Detects non-linear relationships
  - Helps understand feature interactions

### C. Missing Value Patterns
- **What we created:** Bar chart showing missing value counts
- **Why it matters:**
  - Identifies variables with high missingness
  - Helps decide imputation strategy
  - Determines if missingness is random or systematic

---

## 5. KEY INSIGHTS FROM EDA

### Data Quality Issues Identified:
1. **High Missingness:**
   - GCS: 94% missing → Consider excluding or special imputation
   - Past visits: 82% missing → May need to exclude
   - O2 sat: 24% missing → Need imputation strategy

2. **Outliers:**
   - Wait time and length of visit have many outliers (likely real extreme cases)
   - Temperature outliers may represent clinical conditions (hypothermia/hyperthermia)
   - Need to decide: cap outliers, transform, or keep as-is

3. **Class Imbalance:**
   - ESI 3 dominates (47%) while ESI 1 is rare (1.1%)
   - Need stratified sampling and class weights in model

### Feature Insights:
1. **Strong Predictors:**
   - Age (strongest correlation with target)
   - SBP, DBP (blood pressure)
   - GCS (when available)
   - O2 saturation

2. **Multicollinearity:**
   - SBP and DBP highly correlated (0.616) → Consider using only one
   - Pulse and respiration correlated (0.526) → May be redundant

3. **Feature Relationships:**
   - Age negatively correlates with pulse (older = lower pulse)
   - Age positively correlates with BP (older = higher BP)
   - These are expected physiological relationships

### Modeling Recommendations:
1. **Feature Engineering:**
   - Create age groups (bins)
   - Consider BP difference (SBP - DBP) as pulse pressure
   - Handle missing GCS (exclude or impute with mode)

2. **Data Preprocessing:**
   - Impute missing values (median for numerical, mode for categorical)
   - Handle outliers (cap extreme values or use robust scaling)
   - Encode categorical variables

3. **Model Considerations:**
   - Use stratified train/test split
   - Apply class weights to handle imbalance
   - Consider ensemble methods (XGBoost, Random Forest)
   - Use appropriate evaluation metrics (F1-score, precision-recall)

---

## 6. OUTPUT FILES GENERATED

### Visualizations (PNG):
- `univariate_numerical_distributions.png` - Histograms of all numerical variables
- `univariate_numerical_boxplots.png` - Box plots of numerical variables
- `univariate_categorical_barplots.png` - Bar charts of categorical variables
- `esi_level_distribution.png` - Pie chart of target variable
- `outlier_boxplots_detailed.png` - Enhanced box plots with outliers
- `outlier_boxplots_by_esi.png` - Outliers by ESI level
- `outlier_percentage_chart.png` - Outlier percentages visualization
- `bivariate_numerical_correlation_heatmap.png` - Correlation matrix heatmap
- `bivariate_numerical_scatterplots.png` - Top correlation scatter plots
- `bivariate_categorical_stacked_bars.png` - Stacked bar charts
- `bivariate_num_cat_boxplots_esi.png` - Numerical by ESI level box plots
- `multivariate_pairplot.png` - Pair plot of key variables
- `multivariate_target_correlation.png` - Feature correlation with target
- `multivariate_missing_patterns.png` - Missing value patterns

### Data Tables (CSV/Excel):
- `univariate_numerical_summary.csv` - Summary statistics
- `univariate_numerical_skew_kurt.csv` - Skewness and kurtosis
- `univariate_categorical_frequencies.xlsx` - Frequency tables
- `outlier_statistics.csv` - Detailed outlier statistics
- `outlier_summary.csv` - Outlier summary
- `bivariate_numerical_correlation.csv` - Correlation matrix
- `bivariate_categorical_crosstabs.xlsx` - Cross-tabulations
- `bivariate_categorical_chi2_tests.csv` - Chi-square test results
- `bivariate_num_cat_group_stats.xlsx` - Group statistics
- `bivariate_num_cat_anova_tests.csv` - ANOVA test results
- `multivariate_target_correlation.csv` - Target correlations

### Reports:
- `eda_summary_report.txt` - Summary report

---

## Summary

The EDA provided a **comprehensive understanding** of:
1. **Data distributions** - How each variable is distributed
2. **Data quality** - Missing values, outliers, inconsistencies
3. **Relationships** - How variables relate to each other and the target
4. **Feature importance** - Which features are most predictive
5. **Modeling challenges** - Class imbalance, missing data, outliers

This foundation enables informed decisions about:
- Feature selection and engineering
- Data preprocessing strategies
- Model selection and hyperparameter tuning
- Evaluation metric choices

The EDA is complete and ready for the next phase: **Data Preprocessing and Model Training**.

