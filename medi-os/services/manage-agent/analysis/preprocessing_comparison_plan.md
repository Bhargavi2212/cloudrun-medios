# Preprocessing Comparison Analysis Plan

## Objective
Compare data characteristics before and after preprocessing to understand transformations and validate pipeline effectiveness.

## Analysis Components

### 1. Missing Values Analysis
- **Before**: Count missing values per column
- **After**: Verify all missing values are imputed (should be 0)
- **Visualization**: Bar chart comparing missing counts before/after

### 2. Statistical Summary Comparison
- **Before preprocessing**:
  - Mean, median, std dev, min, max for each numeric column
  - Count, unique values, mode for categorical columns
  
- **After preprocessing**:
  - Same statistics on transformed data
  - Note: StandardScaler changes means to ~0, std to ~1
  
- **Visualization**: Side-by-side comparison tables

### 3. Distribution & Skewness Analysis
- **Before**:
  - Calculate skewness for all numeric columns
  - Plot histograms/distributions for each feature
  
- **After**:
  - Calculate skewness after Yeo-Johnson transformation
  - Plot histograms/distributions to show transformation effect
  
- **Visualization**:
  - Before/After skewness bar chart
  - Before/After distribution overlays for key features
  - Q-Q plots to check normality

### 4. Outlier Analysis
- **Before**: Count outliers using IQR method (before clipping)
- **After**: Count outliers after clipping
- **Visualization**: Box plots before/after outlier clipping

### 5. Feature Count & Types
- **Before**: Original feature count and types
- **After**: Final feature count and types (after encoding, dropping, etc.)
- **Visualization**: Feature count comparison chart

### 6. Data Shape Comparison
- **Before**: Original data shape (rows, columns)
- **After**: Train/Val/Test shapes, post-SMOTE shape
- **Visualization**: Data shape comparison chart

### 7. Target Variable Distribution
- **Before**: ESI level distribution (original)
- **After**: ESI level distribution (after SMOTE balancing)
- **Visualization**: Side-by-side bar charts showing class balance improvement

## Implementation Structure

### Script: `compare_preprocessing.py`

**Functions:**
1. `load_raw_data()` - Load CSV before preprocessing
2. `load_preprocessed_data()` - Get preprocessed train/val/test splits
3. `analyze_missing_values()` - Before/after missing value comparison
4. `analyze_distributions()` - Before/after distribution comparison
5. `analyze_skewness()` - Calculate and compare skewness
6. `analyze_outliers()` - Before/after outlier counts
7. `generate_statistical_summary()` - Create comparison tables
8. `create_visualizations()` - Generate all plots
9. `generate_report()` - Create markdown report with findings

**Outputs:**
- `outputs/figures/preprocessing_comparison/` - All plots
- `outputs/preprocessing_comparison_report.md` - Detailed report
- `outputs/preprocessing_statistics.json` - Numerical comparison data

## Visualizations to Generate

1. **Missing Values**
   - Bar chart: Missing count before vs after (should be all zeros after)

2. **Skewness Comparison**
   - Bar chart: Skewness before (colored bars) vs after (overlaid)
   - Highlight which columns were transformed

3. **Distribution Comparison** (Top 10 features)
   - Side-by-side histograms for:
     - Age, SBP, DBP, Pulse, Respiration, Temp, O2 Sat
     - RFV codes (show they remain numeric)
     - Wait time (show Yeo-Johnson transformation effect)

4. **Outlier Analysis**
   - Box plots before/after for top skewed features

5. **Feature Count**
   - Before: Original columns
   - After: Final features after pipeline
   - Show what was added/dropped

6. **Class Distribution**
   - Before: Imbalanced ESI levels
   - After: Balanced ESI levels (after SMOTE)

7. **Correlation Heatmaps**
   - Before: Correlation matrix
   - After: Correlation matrix (note standardization effects)

## Key Metrics to Report

### Missing Values
- Total missing before: X%
- Total missing after: 0% ✅
- Columns with most missing: [list]

### Skewness
- Features with |skew| > 1.0 before: [count]
- Features with |skew| > 1.0 after: [count]
- Improvement: Reduction in skewness

### Outliers
- Total outliers before: [count] (X%)
- Total outliers after: [count] (X%)
- Reduction: X%

### Features
- Original features: 37
- Features after preprocessing: 34
- Added: Cyclical features (4)
- Dropped: Diagnosis columns (3), year (1)

### Class Balance
- Before: ESI 3 dominant (43.6%), ESI 1 rare (1.1%)
- After: All classes balanced to 54,577 samples each
- Improvement: Perfect class balance

## Implementation Order

1. **Create analysis script structure**
2. **Implement data loading functions**
3. **Implement missing value analysis**
4. **Implement skewness analysis**
5. **Implement distribution comparisons**
6. **Implement outlier analysis**
7. **Generate visualizations**
8. **Create comprehensive report**

## Expected Time
- Implementation: ~2-3 hours
- Execution: ~10-15 minutes (includes plotting)

## Success Criteria
- ✅ No missing values after preprocessing
- ✅ Skewness reduced for transformed features
- ✅ Outliers properly clipped
- ✅ Class distribution balanced
- ✅ All visualizations clear and informative

