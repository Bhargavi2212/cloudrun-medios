# Comprehensive Exploratory Data Analysis Report
## NHAMCS Emergency Department Dataset (2011-2022)

**Generated:** 2025-11-02 11:11:07

---

## Executive Summary

- **Total Records:** 178,827
- **Total Variables:** 37
- **Numeric Variables:** 14
- **Categorical Variables:** 16
- **Text Variables:** 6
- **Total Visualizations Generated:** 38

## 1. Data Quality Assessment

- **Data Completeness:** Overall completeness analysis
- **Missing Values:** Check visualizations 01-05
- **Duplicate Records:** 29

## 2. Univariate Analysis Findings

### Target Variable: ESI Level

ESI Level Distribution:
- ESI 0: 5,583 (3.1%)
- ESI 1: 1,878 (1.1%)
- ESI 2: 20,120 (11.3%)
- ESI 3: 77,968 (43.6%)
- ESI 4: 54,729 (30.6%)
- ESI 5: 9,769 (5.5%)
- ESI 7: 8,780 (4.9%)

### Key Findings:

- **Demographics:** Age distribution, gender distribution, temporal patterns
- **Vital Signs:** All vital signs analyzed with distribution plots showing skewness
- **Skewness Analysis:** Distribution plots reveal data skewness (see visualizations 12-13)
- **Categorical Variables:** Binary and multi-category distributions
- **Reason for Visit:** Top reasons analyzed with frequency charts

## 3. Bivariate Analysis Findings

### ESI Level Relationships:
- **Age vs ESI:** Box plots show age distributions by acuity level
- **Vital Signs vs ESI:** Violin plots reveal vital sign patterns by acuity
- **Wait Time vs ESI:** Inverse relationship observed
- **RFV vs ESI:** Different RFV categories associate with different acuity levels

### Correlations:
- **Vital Signs:** Strong correlations between related vitals (e.g., SBP-DBP)
- **Correlation Matrix:** See visualization 27 for full correlation analysis

## 4. Multivariate Analysis Findings

- **Correlation Clustering:** Hierarchical clustering reveals variable groupings
- **Feature Importance:** Random Forest identifies most predictive features
- **Interaction Effects:** Age × Vitals × ESI interactions analyzed
- **Temporal Patterns:** Year and month interactions with ESI levels

## 5. Statistical Test Results

### 5.1 Normality Tests (Shapiro-Wilk)

Tests whether continuous variables follow a normal distribution.

| Variable | Statistic | p-value | Normal? | Skewness | Kurtosis | n |
|----------|-----------|---------|--------|----------|----------|---|
| age | 0.9701 | 0.0000 | **No** | 0.27 | -0.82 | 178,827 |
| dbp | 0.9786 | 0.0000 | **No** | 22.97 | 1074.40 | 178,827 |
| length_of_visit | 0.4887 | 0.0000 | **No** | 6.94 | 75.60 | 150,319 |
| o2_sat | 0.3046 | 0.0000 | **No** | -12.19 | 185.05 | 178,827 |
| pain | 0.9029 | 0.0000 | **No** | -0.16 | -0.99 | 178,827 |
| pulse | 0.5019 | 0.0000 | **No** | 15.34 | 393.10 | 178,827 |
| respiration | 0.4955 | 0.0000 | **No** | 7.01 | 80.37 | 178,827 |
| sbp | 0.9604 | 0.0000 | **No** | 0.80 | 1.91 | 178,827 |
| temp_c | 0.8256 | 0.0000 | **No** | 1.72 | 9.77 | 178,827 |
| wait_time | 0.5599 | 0.0000 | **No** | 7.00 | 80.58 | 178,827 |

### 5.2 Pearson Correlations

Linear correlations between continuous variables.

**Top 10 Significant Correlations:**

| Variables | Correlation | p-value | n |
|-----------|-------------|---------|---|
| sbp dbp | 0.4373 | 0.0000 | 178,827 |
| age sbp | 0.3726 | 0.0000 | 178,827 |
| pulse respiration | 0.3203 | 0.0000 | 178,827 |
| temp c pulse | 0.2480 | 0.0000 | 178,827 |
| age pulse | -0.2446 | 0.0000 | 178,827 |
| age respiration | -0.2211 | 0.0000 | 178,827 |
| age temp c | -0.2144 | 0.0000 | 178,827 |
| wait time length of visit | 0.2012 | 0.0000 | 150,319 |
| temp c respiration | 0.1851 | 0.0000 | 178,827 |
| age length of visit | 0.1357 | 0.0000 | 150,319 |

### 5.3 Spearman Correlations (Non-parametric)

Rank-based correlations (robust to non-normality).

**Top 10 Significant Correlations:**

| Variables | Correlation | p-value | n |
|-----------|-------------|---------|---|
| sbp dbp | 0.6076 | 0.0000 | 178,827 |
| age sbp | 0.3663 | 0.0000 | 178,827 |
| age pulse | -0.3547 | 0.0000 | 178,827 |
| pulse respiration | 0.3456 | 0.0000 | 178,827 |
| age o2 sat | -0.2680 | 0.0000 | 178,827 |
| age length of visit | 0.2486 | 0.0000 | 150,319 |
| age respiration | -0.2483 | 0.0000 | 178,827 |
| temp c pulse | 0.2429 | 0.0000 | 178,827 |
| age temp c | -0.1925 | 0.0000 | 178,827 |
| wait time length of visit | 0.1779 | 0.0000 | 150,319 |

### 5.4 Kruskal-Wallis Tests (ESI Level Differences)

Tests whether continuous variables differ across ESI levels (non-parametric ANOVA).

| Variable | Statistic | p-value | Significant? | Effect Size (η²) | n |
|----------|-----------|---------|--------------|------------------|---|
| age | 10373.71 | 0.0000 | **Yes** | 0.0580 | 178,827 |
| dbp | 406.89 | 0.0000 | **Yes** | 0.0022 | 178,827 |
| gcs | 253.29 | 0.0000 | **Yes** | 0.0080 | 31,084 |
| length_of_visit | 20853.05 | 0.0000 | **Yes** | 0.1387 | 150,319 |
| o2_sat | 1582.99 | 0.0000 | **Yes** | 0.0088 | 178,827 |
| pain | 1410.10 | 0.0000 | **Yes** | 0.0079 | 178,827 |
| pulse | 290.50 | 0.0000 | **Yes** | 0.0016 | 178,827 |
| respiration | 282.46 | 0.0000 | **Yes** | 0.0015 | 178,827 |
| sbp | 1301.78 | 0.0000 | **Yes** | 0.0072 | 178,827 |
| temp_c | 124.95 | 0.0000 | **Yes** | 0.0007 | 178,827 |
| wait_time | 1098.16 | 0.0000 | **Yes** | 0.0061 | 178,827 |

### 5.5 One-Way ANOVA Tests (ESI Level Differences)

Parametric alternative to Kruskal-Wallis (assumes normality).

| Variable | F-statistic | p-value | Significant? | Effect Size (η²) | n |
|----------|-------------|---------|--------------|------------------|---|
| age | 1855.77 | 0.0000 | **Yes** | 0.0586 | 178,827 |
| pulse | 77.28 | 0.0000 | **Yes** | 0.0026 | 178,827 |
| temp_c | 17.19 | 0.0000 | **Yes** | 0.0006 | 178,827 |
| wait_time | 60.74 | 0.0000 | **Yes** | 0.0020 | 178,827 |

### 5.6 Chi-Square Tests (Categorical Associations)

Tests associations between ESI level and categorical variables.

| Variables | χ² | p-value | Significant? | Cramér's V | n |
|-----------|----|---------|--------------|------------|---|
| esi level vs ambulance arrival | 10921.04 | 0.0000 | **Yes** | 0.1747 | 178,827 |
| esi level vs cebvd | 2082.81 | 0.0000 | **Yes** | 0.1079 | 178,827 |
| esi level vs chf | 2669.78 | 0.0000 | **Yes** | 0.1222 | 178,827 |
| esi level vs diabetes | 687.03 | 0.0000 | **Yes** | 0.1162 | 50,916 |
| esi level vs discharged 7d | 254.63 | 0.0000 | **Yes** | 0.0640 | 31,084 |
| esi level vs hiv | 143.07 | 0.0000 | **Yes** | 0.0283 | 178,827 |
| esi level vs injury | 3833.33 | 0.0000 | **Yes** | 0.1035 | 178,827 |
| esi level vs no chronic conditions | 9230.19 | 0.0000 | **Yes** | 0.1606 | 178,827 |
| esi level vs seen 72h | 1519.04 | 0.0000 | **Yes** | 0.0652 | 178,827 |
| esi level vs sex | 511.73 | 0.0000 | **Yes** | 0.0535 | 178,827 |

### 5.7 Mann-Whitney U Tests (Binary vs Continuous)

Tests differences in continuous variables between binary groups.

| Comparison | Statistic | p-value | Significant? | Effect Size (r) |
|------------|-----------|---------|--------------|------------------|
| age by ambulance arrival | 56430764.00 | 0.0000 | **Yes** | 0.2847 |
| age by discharged 7d | 4562031.00 | 0.0000 | **Yes** | 0.1748 |
| age by injury | 3496627542.50 | 0.0000 | **Yes** | -0.0148 |
| age by seen 72h | 56826003.00 | 0.1595 | No | 0.0114 |
| length of visit by ambulance arrival | 44868170.00 | 0.0000 | **Yes** | 0.2400 |
| length of visit by discharged 7d | 4638265.50 | 0.0000 | **Yes** | 0.1610 |
| length of visit by injury | 2769919041.50 | 0.0000 | **Yes** | -0.1368 |
| length of visit by seen 72h | 42325837.50 | 0.0000 | **Yes** | 0.0518 |
| pain by ambulance arrival | 85667159.00 | 0.0000 | **Yes** | -0.0859 |
| pain by discharged 7d | 5578162.00 | 0.6498 | No | -0.0090 |
| pain by injury | 3225353658.00 | 0.0000 | **Yes** | 0.0639 |
| pain by seen 72h | 58679730.50 | 0.0086 | **Yes** | -0.0208 |
| wait time by ambulance arrival | 96704207.50 | 0.0000 | **Yes** | -0.2258 |
| wait time by discharged 7d | 5656291.00 | 0.2525 | No | -0.0232 |
| wait time by injury | 3557460151.00 | 0.0000 | **Yes** | -0.0325 |
| wait time by seen 72h | 55399906.00 | 0.0000 | **Yes** | 0.0362 |

## 6. Modeling Recommendations

### Feature Engineering:
1. **Handle Skewness:** Apply log or Box-Cox transformations to highly skewed variables
2. **Text Features:** Use TF-IDF or embeddings for RFV text variables
3. **Temporal Features:** Create cyclical features for month/day-of-week
4. **Interaction Terms:** Consider Age × Vitals, Comorbidities × RFV interactions
5. **Missing Data:** Use median imputation for vitals (already implemented) or advanced imputation

### Preprocessing:
1. **Standardization:** Standardize numeric features for distance-based algorithms
2. **Class Imbalance:** Address ESI level imbalance (see distribution)
3. **Outlier Handling:** Consider IQR-based outlier detection for vital signs
4. **Feature Selection:** Use feature importance rankings to reduce dimensionality

### Model Recommendations:
1. **Primary Model:** Gradient Boosting (XGBoost/LightGBM) - handles non-linear relationships
2. **Alternative:** Random Forest - interpretable, handles mixed data types
3. **Neural Network:** Consider deep learning for capturing complex interactions
4. **Ensemble:** Combine multiple models for improved performance

### Class Imbalance Strategy:
1. **Resampling:** Use SMOTE or ADASYN for minority ESI classes
2. **Class Weights:** Apply inverse frequency class weights
3. **Cost-Sensitive Learning:** Penalize misclassification of critical cases (ESI 1-2)

### Validation Strategy:
1. **Temporal Split:** Use chronological split (train on earlier years, test on later)
2. **Stratified K-Fold:** Ensure ESI level distribution maintained in folds
3. **Metrics:** Use macro-averaged F1 (handles imbalance) and weighted accuracy
4. **Clinical Validation:** Review misclassifications with domain experts

### Feature Priority (Based on Importance Analysis):
Top features to prioritize:
- length_of_visit: 0.1308
- age: 0.1045
- pulse: 0.1033
- wait_time: 0.1020
- sbp: 0.0982
- dbp: 0.0931
- temp_c: 0.0917
- respiration: 0.0579
- o2_sat: 0.0562
- pain: 0.0443

---

## Visualizations Index

All 38 visualizations are saved in `figures/` directory.
Refer to figure numbers in filenames for specific analyses.

