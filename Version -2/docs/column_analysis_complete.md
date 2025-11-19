# Complete Column Analysis for NHAMCS Triage Dataset

## Dataset Overview
- **Total Records**: 163,736
- **Total Columns**: 21
- **Duplicates**: 0 (no duplicate rows)

---

## Column-by-Column Detailed Analysis

### 1. **year** (Metadata - EXCLUDE)
- **Type**: Integer (int64)
- **Description**: Year of data collection (2011-2022)
- **Missing**: 0% (0 nulls)
- **Unique Values**: 11 (2011-2022)
- **Mean**: 2015.87
- **Skewness**: 0.14 (normal)
- **Outliers**: 0%
- **Preprocessing**: **EXCLUDE** - This is metadata/identifier, not a feature
- **Reason to Exclude**: Not predictive, just identifies data source year

---

### 2. **line_num** (Metadata - EXCLUDE)
- **Type**: Integer (int64)
- **Description**: Line number from original data file
- **Missing**: 0% (0 nulls)
- **Unique Values**: 30,720
- **Mean**: 10,903.83
- **Skewness**: 0.50 (normal)
- **Outliers**: 0%
- **Preprocessing**: **EXCLUDE** - This is an identifier, not a feature
- **Reason to Exclude**: Row identifier, not predictive

---

### 3. **esi_level** (TARGET VARIABLE)
- **Type**: Float (float64) - but actually ordinal categorical
- **Description**: Emergency Severity Index (1-5)
  - ESI 1: Most critical (resuscitation)
  - ESI 2: Very urgent
  - ESI 3: Urgent
  - ESI 4: Less urgent
  - ESI 5: Least urgent
- **Missing**: 0% (0 nulls)
- **Unique Values**: 5 (1, 2, 3, 4, 5)
- **Mean**: 3.31
- **Median**: 3.0
- **Std**: 0.80
- **Range**: 1.0 - 5.0
- **Distribution**: 
  - ESI 1: 1,866 (1.1%)
  - ESI 2: 19,997 (12.2%)
  - ESI 3: 77,629 (47.4%) ← Most common
  - ESI 4: 54,519 (33.3%)
  - ESI 5: 9,725 (5.9%)
- **Skewness**: -0.05 (normal)
- **Variable Type**: **ORDINAL** (ordered categories)
- **Preprocessing**: **NO PREPROCESSING NEEDED** - This is the target variable
- **Note**: Severe class imbalance (ESI 3 = 47%, ESI 1 = 1.1%)

---

### 4. **pulse** (Vital Sign)
- **Type**: Continuous Numerical (float64)
- **Description**: Heart rate in beats per minute
- **Missing**: **5.20%** (8,511 nulls) ⚠️ **>5% - Consider excluding**
- **Unique Values**: 219
- **Mean**: 91.35 bpm
- **Median**: 88.0 bpm
- **Std**: 22.97
- **Range**: 1.0 - 244.0 bpm
- **Q25**: 76, Q75**: 102
- **Normal Range**: 60-100 bpm
- **Skewness**: 1.10 ⚠️ **Highly skewed - needs transformation**
- **Outliers**: 6,045 (3.89%) - Moderate
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: Median imputation (if keeping column)
  - **Transformation**: Log or Box-Cox transformation (skewness > 1)
  - **Outliers**: Cap at reasonable bounds (e.g., 40-200 bpm)
- **Decision**: Since >5% missing, consider excluding OR impute with median

---

### 5. **respiration** (Vital Sign)
- **Type**: Continuous Numerical (float64)
- **Description**: Respiratory rate in breaths per minute
- **Missing**: 3.05% (4,988 nulls) ✓ **<5% - Keep**
- **Unique Values**: 58
- **Mean**: 19.23 bpm
- **Median**: 18.0 bpm
- **Std**: 4.58
- **Range**: 1.0 - 59.0 bpm
- **Q25**: 16, Q75**: 20
- **Normal Range**: 12-20 bpm
- **Skewness**: 3.11 ⚠️ **Highly skewed - needs transformation**
- **Outliers**: 9,084 (5.72%) - Moderate
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: Median imputation (3.05% is acceptable)
  - **Transformation**: Log or Box-Cox transformation (skewness = 3.11)
  - **Outliers**: Cap at reasonable bounds (e.g., 8-40 bpm)

---

### 6. **sbp** (Systolic Blood Pressure - Vital Sign)
- **Type**: Continuous Numerical (float64)
- **Description**: Systolic blood pressure in mmHg
- **Missing**: **10.01%** (16,396 nulls) ⚠️ **>5% - Consider excluding**
- **Unique Values**: 221
- **Mean**: 134.23 mmHg
- **Median**: 132.0 mmHg
- **Std**: 23.73
- **Range**: 12.0 - 290.0 mmHg
- **Q25**: 118, Q75**: 147
- **Normal Range**: 90-140 mmHg
- **Skewness**: 0.77 (moderately skewed)
- **Outliers**: 3,697 (2.51%) - Low
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: Median imputation (if keeping) OR exclude if >10% missing
  - **Transformation**: May need transformation (skewness = 0.77)
  - **Outliers**: Cap at reasonable bounds (e.g., 70-250 mmHg)
- **Decision**: Since >10% missing, consider excluding OR impute with median

---

### 7. **dbp** (Diastolic Blood Pressure - Vital Sign)
- **Type**: Continuous Numerical (float64)
- **Description**: Diastolic blood pressure in mmHg
- **Missing**: **10.55%** (17,275 nulls) ⚠️ **>5% - Consider excluding**
- **Unique Values**: 156
- **Mean**: 78.37 mmHg
- **Median**: 78.0 mmHg
- **Std**: 14.68
- **Range**: 11.0 - 190.0 mmHg
- **Q25**: 69, Q75**: 87
- **Normal Range**: 60-90 mmHg
- **Skewness**: 0.44 (normal)
- **Outliers**: 2,711 (1.85%) - Low
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: Median imputation (if keeping) OR exclude if >10% missing
  - **Transformation**: Not needed (skewness = 0.44)
  - **Outliers**: Cap at reasonable bounds (e.g., 40-120 mmHg)
- **Decision**: Since >10% missing, consider excluding OR impute with median
- **Note**: Highly correlated with SBP (0.616) - may be redundant

---

### 8. **o2_sat** (Oxygen Saturation - Vital Sign)
- **Type**: Continuous Numerical (float64)
- **Description**: Oxygen saturation percentage
- **Missing**: **23.93%** (39,184 nulls) ⚠️ **>5% - Consider excluding**
- **Unique Values**: 84
- **Mean**: 97.16%
- **Median**: 98.0%
- **Std**: 5.41
- **Range**: 1.0 - 100.0%
- **Q25**: 96, Q75**: 100
- **Normal Range**: 95-100%
- **Skewness**: -11.20 ⚠️ **Extremely skewed - needs transformation**
- **Outliers**: 2,588 (2.08%) - Low
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: Median imputation (if keeping) OR exclude (>20% missing)
  - **Transformation**: Strong transformation needed (skewness = -11.20)
  - **Outliers**: Cap at reasonable bounds (e.g., 70-100%)
- **Decision**: Since >20% missing, consider excluding OR impute with median

---

### 9. **gcs** (Glasgow Coma Scale - Vital Sign)
- **Type**: Ordinal Numerical (float64)
- **Description**: Glasgow Coma Scale score (3-15)
  - 3 = worst (deep coma)
  - 15 = best (fully conscious)
- **Missing**: **94.42%** (154,595 nulls) ⚠️ **>5% - EXCLUDE**
- **Unique Values**: 12 (3-15)
- **Mean**: 14.73 (when present)
- **Median**: 15.0
- **Std**: 1.57
- **Range**: 3.0 - 15.0
- **Skewness**: -6.44 ⚠️ **Extremely skewed**
- **Outliers**: 451 (4.93%) - Low
- **Variable Type**: **ORDINAL** numerical
- **Preprocessing Strategy**: **EXCLUDE** - 94% missing is too high
- **Decision**: **DO NOT USE** - Too much missing data

---

### 10. **pain** (Pain Scale - Clinical)
- **Type**: Ordinal Numerical (float64)
- **Description**: Pain scale score (0-10)
  - 0 = no pain
  - 10 = severe pain
- **Missing**: **22.86%** (37,437 nulls) ⚠️ **>5% - Consider excluding**
- **Unique Values**: 11 (0-10)
- **Mean**: 4.69
- **Median**: 5.0
- **Std**: 3.71
- **Range**: 0.0 - 10.0
- **Skewness**: -0.07 (normal)
- **Outliers**: 0 (0%) - None
- **Variable Type**: **ORDINAL** numerical
- **Preprocessing Strategy**:
  - **Missing**: Median imputation (if keeping) OR exclude (>20% missing)
  - **Transformation**: Not needed (skewness = -0.07)
  - **Outliers**: None
- **Decision**: Since >20% missing, consider excluding OR impute with median

---

### 11. **age** (Demographic)
- **Type**: Continuous Numerical (float64)
- **Description**: Patient age in years
- **Missing**: 0% (0 nulls) ✓ **Perfect**
- **Unique Values**: 95
- **Mean**: 38.38 years
- **Median**: 36.0 years
- **Std**: 24.20
- **Range**: 0.0 - 94.0 years
- **Q25**: 20, Q75**: 56
- **Skewness**: 0.27 (normal)
- **Outliers**: 0 (0%) - None
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: None (0% missing)
  - **Transformation**: Not needed (skewness = 0.27)
  - **Binning**: Consider creating age groups (0-18, 19-35, 36-50, 51-65, 65+)
  - **Outliers**: None
- **Note**: Strong predictor (correlation with ESI = -0.239)

---

### 12. **month** (Temporal)
- **Type**: Ordinal Numerical (float64)
- **Description**: Month of visit (1-12)
- **Missing**: 0% (0 nulls) ✓ **Perfect**
- **Unique Values**: 12 (1-12)
- **Mean**: 6.33
- **Median**: 6.0
- **Std**: 3.43
- **Range**: 1.0 - 12.0
- **Skewness**: 0.04 (normal)
- **Outliers**: 0 (0%) - None
- **Variable Type**: **ORDINAL** numerical, **CYCLIC** (12 → 1)
- **Preprocessing Strategy**:
  - **Missing**: None (0% missing)
  - **Encoding**: 
    - Option 1: Cyclic encoding (sin/cos transformation)
    - Option 2: Ordinal encoding (1-12)
    - Option 3: One-hot encoding (12 categories)
  - **Transformation**: Not needed
- **Note**: Cyclic nature (December → January) - cyclic encoding recommended

---

### 13. **wait_time** (Operational)
- **Type**: Continuous Numerical (float64)
- **Description**: Wait time in minutes
- **Missing**: 0% (0 nulls) ✓ **Perfect**
- **Unique Values**: 816
- **Mean**: 34.50 minutes
- **Median**: 15.0 minutes
- **Std**: 67.82
- **Range**: -9.0 - 1440.0 minutes
- **Q25**: 4, Q75**: 39
- **Skewness**: 6.58 ⚠️ **Extremely skewed - needs transformation**
- **Outliers**: 15,554 (9.50%) ⚠️ **High outlier percentage**
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: Handle -9 values (missing code) → replace with median
  - **Transformation**: Log transformation (skewness = 6.58)
  - **Outliers**: Cap at reasonable bounds (e.g., 0-300 minutes)
- **Note**: Many outliers (9.5%) - needs robust handling

---

### 14. **length_of_visit** (Operational)
- **Type**: Continuous Numerical (float64)
- **Description**: Length of visit in minutes
- **Missing**: **15.82%** (25,904 nulls) ⚠️ **>5% - Consider excluding**
- **Unique Values**: 2,357
- **Mean**: 226.68 minutes
- **Median**: 158.0 minutes
- **Std**: 301.33
- **Range**: -9.0 - 5741.0 minutes
- **Q25**: 89, Q75**: 264
- **Skewness**: 6.69 ⚠️ **Extremely skewed - needs transformation**
- **Outliers**: 8,875 (6.44%) - Moderate
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: Median imputation (if keeping) OR exclude (>15% missing)
  - **Transformation**: Log transformation (skewness = 6.69)
  - **Outliers**: Cap at reasonable bounds (e.g., 0-600 minutes)
- **Decision**: Since >15% missing, consider excluding OR impute with median

---

### 15. **ambulance_arrival** (Clinical - Binary)
- **Type**: Binary Categorical (float64)
- **Description**: Whether patient arrived by ambulance
- **Missing**: 0% (but has -9/-8 codes = missing)
- **Unique Values**: 4 (-9, -8, 1, 2)
- **Value Codes**:
  - 1 = Yes (arrived by ambulance)
  - 2 = No (did not arrive by ambulance)
  - -9, -8 = Missing/Unknown
- **Distribution**: 
  - 1 (Yes): 27,013 (16.5%)
  - 2 (No): 132,301 (80.8%)
  - -9/-8: 4,422 (2.7%)
- **Skewness**: -5.44 ⚠️ **Highly skewed** (due to binary nature)
- **Variable Type**: **NOMINAL** categorical (binary)
- **Preprocessing Strategy**:
  - **Missing**: Replace -9/-8 with mode (2 = No) OR create separate "unknown" category
  - **Encoding**: Binary encoding (0 = No, 1 = Yes)
  - **Transformation**: Not needed (binary variable)

---

### 16. **seen_72h** (History - Binary)
- **Type**: Binary Categorical (float64)
- **Description**: Whether patient was seen in last 72 hours
- **Missing**: 0% (but has -9/-8 codes = missing)
- **Unique Values**: 4 (-9, -8, 1, 2)
- **Value Codes**:
  - 1 = Yes (seen in last 72h)
  - 2 = No (not seen in last 72h)
  - -9, -8 = Missing/Unknown
- **Distribution**:
  - 1 (Yes): 6,589 (4.0%)
  - 2 (No): 143,913 (87.9%)
  - -9/-8: 13,234 (8.1%)
- **Skewness**: -3.05 ⚠️ **Highly skewed** (due to binary nature)
- **Variable Type**: **NOMINAL** categorical (binary)
- **Preprocessing Strategy**:
  - **Missing**: Replace -9/-8 with mode (2 = No) OR create separate "unknown" category
  - **Encoding**: Binary encoding (0 = No, 1 = Yes)
  - **Transformation**: Not needed (binary variable)

---

### 17. **discharged_7d** (History - Binary)
- **Type**: Binary Categorical (float64)
- **Description**: Whether patient was discharged in last 7 days
- **Missing**: **81.58%** (133,580 nulls) ⚠️ **>5% - EXCLUDE**
- **Unique Values**: 4 (-9, -8, 1, 2)
- **Value Codes**:
  - 1 = Yes
  - 2 = No
  - -9, -8 = Missing
- **Distribution** (of non-missing):
  - 1 (Yes): 843 (2.8%)
  - 2 (No): 16,780 (55.6%)
  - -9/-8: 12,533 (41.6%)
- **Skewness**: -0.34 (normal)
- **Variable Type**: **NOMINAL** categorical (binary)
- **Preprocessing Strategy**: **EXCLUDE** - 81.58% missing is too high
- **Decision**: **DO NOT USE** - Too much missing data

---

### 18. **past_visits** (History)
- **Type**: Discrete Numerical (float64)
- **Description**: Number of past visits
- **Missing**: **81.58%** (133,580 nulls) ⚠️ **>5% - EXCLUDE**
- **Unique Values**: 26 (0-23)
- **Mean**: -1.88 (when including -9/-8 codes)
- **Median**: 0.0 (of valid values)
- **Range**: -9 to 23
- **Skewness**: 0.46 (normal)
- **Outliers**: 223 (0.74%) - Low
- **Variable Type**: Discrete numerical
- **Preprocessing Strategy**: **EXCLUDE** - 81.58% missing is too high
- **Decision**: **DO NOT USE** - Too much missing data

---

### 19. **injury** (Clinical - Binary)
- **Type**: Binary Categorical (int64)
- **Description**: Whether visit is injury-related
- **Missing**: 0% (0 nulls) ✓ **Perfect**
- **Unique Values**: 2 (0, 1)
- **Distribution**:
  - 0 (No): 121,070 (73.9%)
  - 1 (Yes): 42,666 (26.1%)
- **Skewness**: 1.09 (moderately skewed due to binary nature)
- **Variable Type**: **NOMINAL** categorical (binary)
- **Preprocessing Strategy**:
  - **Missing**: None (0% missing)
  - **Encoding**: Already binary (0/1) - no encoding needed
  - **Transformation**: Not needed

---

### 20. **temp_c** (Temperature - Vital Sign)
- **Type**: Continuous Numerical (float64)
- **Description**: Body temperature in Celsius
- **Missing**: 4.39% (7,192 nulls) ✓ **<5% - Keep**
- **Unique Values**: 172
- **Mean**: 36.82°C
- **Median**: 36.78°C
- **Std**: 0.58
- **Range**: 30.22 - 43.28°C
- **Q25**: 36.56, Q75**: 37.0
- **Normal Range**: 36.1-37.2°C
- **Skewness**: 1.71 ⚠️ **Highly skewed - needs transformation**
- **Outliers**: 11,777 (7.52%) - Moderate
- **Variable Type**: Continuous numerical
- **Preprocessing Strategy**:
  - **Missing**: Median imputation (4.39% is acceptable)
  - **Transformation**: Log or Box-Cox transformation (skewness = 1.71)
  - **Outliers**: Cap at reasonable bounds (e.g., 30-42°C)

---

### 21. **rfv1_cluster** (Reason for Visit - Categorical)
- **Type**: Nominal Categorical (object/string)
- **Description**: Reason for visit cluster (13 categories)
- **Missing**: 0.09% (149 nulls) ✓ **<5% - Keep**
- **Unique Values**: 13 categories
- **Distribution**:
  - Musculoskeletal: 34,084 (20.8%)
  - Other: 26,718 (16.3%)
  - Gastrointestinal: 21,250 (13.0%)
  - Respiratory: 14,035 (8.6%)
  - Neurological: 14,034 (8.6%)
  - Trauma_Injury: 12,578 (7.7%)
  - Cardiovascular: 10,686 (6.5%)
  - Mental_Health: 7,159 (4.4%)
  - Fever_Infection: 6,490 (4.0%)
  - Skin: 5,760 (3.5%)
  - Ear_Nose_Throat: 5,329 (3.3%)
  - Urinary_Genitourinary: 2,775 (1.7%)
  - General_Symptoms: 2,689 (1.6%)
- **Variable Type**: **NOMINAL** categorical (no order)
- **Preprocessing Strategy**:
  - **Missing**: Mode imputation (replace 149 nulls with "Other")
  - **Encoding**: 
    - Option 1: One-hot encoding (13 binary columns)
    - Option 2: Target encoding (mean ESI per cluster)
    - Option 3: Label encoding (not recommended for nominal)
  - **Transformation**: Not needed

---

## Chronic Condition Columns (Binary - Already Encoded)

These columns are already binary (0/1) and have 0% missing:
- **cebvd**: Cerebrovascular disease (0/1)
- **chf**: Congestive heart failure (0/1)
- **ed_dialysis**: ED dialysis (0/1)
- **hiv**: HIV (0/1)
- **diabetes**: Diabetes (0/1)
- **no_chronic_conditions**: No chronic conditions (0/1)

**Preprocessing**: No preprocessing needed - already binary encoded

---

## Summary of Preprocessing Decisions

### Columns to EXCLUDE (>5% missing or metadata):
1. **year** - Metadata
2. **line_num** - Identifier
3. **gcs** - 94.42% missing
4. **discharged_7d** - 81.58% missing
5. **past_visits** - 81.58% missing
6. **pulse** - 5.20% missing (borderline - can keep with imputation)
7. **sbp** - 10.01% missing
8. **dbp** - 10.55% missing
9. **o2_sat** - 23.93% missing
10. **pain** - 22.86% missing
11. **length_of_visit** - 15.82% missing

### Columns to KEEP (with preprocessing):
1. **esi_level** - Target (no preprocessing)
2. **respiration** - 3.05% missing → Median imputation + transformation
3. **age** - 0% missing → Consider binning
4. **month** - 0% missing → Cyclic encoding
5. **wait_time** - 0% missing → Handle -9 codes + log transformation
6. **ambulance_arrival** - Handle -9/-8 codes → Binary encoding
7. **seen_72h** - Handle -9/-8 codes → Binary encoding
8. **injury** - 0% missing → Already binary
9. **temp_c** - 4.39% missing → Median imputation + transformation
10. **rfv1_cluster** - 0.09% missing → Mode imputation + one-hot encoding
11. **Chronic conditions** (6 columns) - Already binary, no preprocessing

---

## Preprocessing Strategy Summary

### 1. Data Split
- **First step**: Split into Train (70%) / Validation (15%) / Test (15%)
- **Stratified split** by ESI level (to handle class imbalance)

### 2. Missing Value Handling
- **>5% missing**: Exclude column (gcs, discharged_7d, past_visits, sbp, dbp, o2_sat, pain, length_of_visit)
- **<5% missing**: Impute with median (numerical) or mode (categorical)
- **Special codes (-9, -8)**: Replace with appropriate imputation value

### 3. Skewness Transformation
- **Highly skewed (|skew| > 1)**: Log or Box-Cox transformation
  - pulse, respiration, o2_sat, wait_time, length_of_visit, temp_c
- **Moderately skewed (0.5 < |skew| < 1)**: May need transformation
  - sbp

### 4. Outlier Treatment
- **High outliers (>10%)**: Cap at reasonable bounds
  - wait_time (9.5%)
- **Moderate outliers (5-10%)**: Cap or robust scaling
  - temp_c (7.5%), length_of_visit (6.4%), respiration (5.7%)

### 5. Encoding
- **Ordinal variables**: Ordinal encoding
  - esi_level (target), month (cyclic), pain, gcs (if kept)
- **Nominal variables**: One-hot or target encoding
  - rfv1_cluster (one-hot recommended)
- **Binary variables**: Already 0/1 or binary encoding
  - ambulance_arrival, seen_72h, injury, chronic conditions

### 6. Feature Engineering
- **Age**: Consider binning into groups
- **Month**: Cyclic encoding (sin/cos)
- **BP**: Consider pulse pressure (SBP - DBP) if both kept

---

## Final Feature List (After Exclusions)

**Numerical Features (with preprocessing)**:
1. respiration (impute + transform)
2. age (consider binning)
3. wait_time (handle -9 + transform)
4. temp_c (impute + transform)

**Categorical Features (with encoding)**:
1. month (cyclic encoding)
2. rfv1_cluster (one-hot encoding)
3. ambulance_arrival (binary encoding)
4. seen_72h (binary encoding)
5. injury (already binary)

**Binary Features (no preprocessing)**:
1. cebvd, chf, ed_dialysis, hiv, diabetes, no_chronic_conditions

**Total Features**: ~25-30 (after one-hot encoding of rfv1_cluster)

---

## Next Steps

1. **Split data** first (train/val/test)
2. **Fit preprocessing** on training set only
3. **Transform** validation and test sets using fitted transformers
4. **Handle missing values** based on strategy above
5. **Apply transformations** for skewed features
6. **Encode categorical** variables
7. **Handle outliers** with capping or robust scaling

