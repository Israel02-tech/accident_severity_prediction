# Comprehensive Technical & Experimental Report: Road Traffic Accident Severity Prediction Using Explainable Machine Learning

**Project Title:** The Development of an Artificial Intelligence / Machine Learning Model for Identifying Patterns in Road Traffic Accidents  
**Author:** Progress Comfort Onyekachi (Matriculation No: E051555)  
**Department:** Department of Computing, Distance Learning Centre, University of Ibadan  
**Supervisor:** Dr. B. I. Ayinla  

---

## 1. Executive Summary

This report documents the complete implementation, experimental workflow, statistical checks, model evaluations, explainability stability tests, and road-safety policy synthesis carried out for the B.Sc. Computer Science research project on road traffic crash severity prediction.

Following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology, a leakage-safe, group-aware machine learning pipeline was developed and evaluated on **12,316 police-recorded road traffic crashes** from Addis Ababa, Ethiopia (`RTA Dataset.csv`, 32 attributes). Crash severity was modeled primarily as an imbalanced **three-class classification task**—**Slight Injury** (`84.56%`), **Serious Injury** (`14.15%`), and **Fatal Injury** (`1.28%`, a `65.9 : 1` slight-to-fatal imbalance ratio)—supplemented by a **binary Killed-or-Seriously-Injured (KSI)** benchmark (`Fatal + Serious = 15.44%` vs. `Slight = 84.56%`).

### Key Headline Findings
1. **Data Quality & Leakage Discovery**:
   - Seven post-crash casualty outcome attributes (`Number_of_casualties`, `Casualty_class`, `Casualty_severity`, `Sex_of_casualty`, `Age_band_of_casualty`, `Work_of_casuality`, `Fitness_of_casuality`) were excluded to prevent post-event target leakage.
   - Crucially, exploratory auditing revealed that **6,221 of the 12,316 records (`50.5%`)** occur immediately adjacent to another record with the exact same `(Day_of_week, Time)` timestamp—with **`97.69%` identical crash severity**—because multi-vehicle crashes were logged as multiple vehicle/driver rows. Assigning a `crash_event_id` and partitioning via **Group-Aware Stratified Splitting (`70% Train : 15% Validation : 15% Test`)** eliminated `100%` of multi-row crash leakage across splits.
2. **Research Question 1 (Imbalance Handling)**:
   - Under **No Balancing**, models achieved deceptive overall accuracy (`83.4%–85.0%`) by predicting *Slight Injury* almost exclusively, resulting in **`0.00%` Fatal Recall** across all three tested classifiers and `0.00%–3.80%` Serious Recall.
   - **Class Weighting** (cost-sensitive learning) delivered the highest overall **5-fold CV Macro-F1 (`0.3553 ± 0.0229` with XGBoost)**, lifting Serious Recall from `1.01%` to **`33.63%`** and Fatal Recall from `0.00%` to **`13.67%`** without fabricating synthetic categorical records. **SMOTE-NC** ranked a close second (`0.3517 ± 0.0187` with XGBoost; `0.3364 ± 0.0052` with Random Forest).
3. **Research Question 2 (Algorithm Comparison)**:
   - Across **10 repeated group-stratified CV folds**, the **15% Validation split**, the **Binary KSI benchmark**, and the untouched **15% Held-Out Test Set (`1,829` crashes)**, ensemble methods substantially outperformed the single **Decision Tree baseline**:
     - **Random Forest (Champion)**: Held-out Test **Macro-F1 = `0.3844`** (`69.22%` Accuracy, `76.21%` Slight Recall, `33.57%` Serious Recall, `20.00%` Fatal Recall; Repeated CV Macro-F1 = `0.3621 ± 0.0251`; Binary KSI Macro-F1 = `0.5603`).
     - **XGBoost (Runner-Up)**: Held-out Test **Macro-F1 = `0.3774`**, highest **Serious Recall (`41.07%`)**, and highest **Macro OvR ROC-AUC (`0.5694`)** (Repeated CV Macro-F1 = `0.3552 ± 0.0155`; Binary KSI Macro-F1 = `0.5469`).
     - **Multilayer Perceptron (MLP)**: Held-out Test **Macro-F1 = `0.3499`** (`72.94%` Accuracy, `21.79%` Serious Recall, `0.00%` Fatal Recall; exhibited strong training memorization `0.9739` vs. `0.3544` CV despite L2 regularization and early stopping).
     - **Decision Tree (Baseline)**: Held-out Test **Macro-F1 = `0.3082`** (`49.54%` Accuracy; over-predicted minority classes at the expense of precision).
4. **Research Question 3 (Explainability Stability & Haddon Matrix)**:
   - Exact additive **TreeSHAP** attributions aggregated across the `157` encoded columns back to the `27` parent crash attributes demonstrated **strong cross-fold stability** across 5 independent group-stratified CV folds: **mean pairwise Spearman rank correlation $\rho = 0.7875$** and **mean Top-10 Jaccard overlap $J = 0.7296$**.
   - Exactly **9 predictors** passed the strict stability threshold (appearing in the Top 10 in $\ge 4/5$ folds): **`Types_of_Junction`** (ranked **#1 in all 5/5 folds**), **`Is_weekend`** (5/5), **`Day_of_week`** (5/5), **`Number_of_vehicles_involved`** (5/5), **`Time_of_day`** (5/5), **`Hour`** (5/5), **`Age_band_of_driver`** (5/5), **`Cause_of_accident`** (4/5), and **`Lanes_or_Medians`** (4/5).
   - These 9 verified stable factors were mapped systematically into the **Haddon Matrix** (*Human*, *Vehicle*, *Road/Environment* × *Pre-Crash*, *Crash*, *Post-Crash*) and deployed in an interactive **Streamlit** decision-support prototype (`app.py`).

---

## 2. Complete End-to-End Experimental Workflow

### 2.1 Software Environment & Reproducibility Setup
All experiments were executed in an isolated Python `3.9.6` virtual environment (`.venv`) with fixed random seeds (`random_state = 42`) and pinned package versions in `requirements.txt`:
- **Data Processing & Statistics**: `numpy==2.0.2`, `pandas==2.3.3`, `scipy==1.13.1`
- **Machine Learning & Imbalance Handling**: `scikit-learn==1.6.1`, `xgboost==2.1.4` (with OpenMP `libomp`), `imbalanced-learn==0.12.4`, `Boruta==0.4.3`, `joblib==1.5.3`
- **Explainable AI, Visualization & Deployment**: `shap==0.49.1`, `matplotlib==3.9.4`, `seaborn==0.13.2`, `jupyter==1.1.1`, `streamlit==1.50.0`

### 2.2 Pipeline Architecture Across Project Files
- **`src/utils.py`**: Shared constants, text cleaning (`clean_rta_data`), event ID assignment (`assign_crash_event_id`), custom `scikit-learn` transformers (`TemporalFeatureExtractor`, `OrdinalCategoryMapper`), leakage-safe `70:15:15` splitter (`group_stratified_split`), and multi-class evaluation metrics (`evaluate_multiclass`).
- **`notebooks/01_eda_and_data_cleaning.ipynb`**: Schema audit, leakage removal, text repair, missing-value audit, event-grouping verification, exploratory charts, and Pearson Chi-Square ($\chi^2$) + Cramér's V feature association tests. Outputs `data/processed/rta_cleaned.csv`.
- **`notebooks/02_preprocessing_and_imbalance_rq1.ipynb`**: Group-stratified `70:15:15` splitting, temporal feature extraction, Spearman multicollinearity check, Boruta all-relevant feature selection, naive-vs-group split check, and the 5-fold RQ1 comparison (`No Balancing`, `Class Weighting`, `SMOTE-NC`).
- **`notebooks/03_model_training_and_evaluation_rq2.ipynb`**: Hyperparameter tuning (`ParameterSampler` over 5-fold Group-Stratified CV optimizing Macro-F1), 10-fold repeated CV model comparison (RQ2), Validation check, Binary KSI fallback comparison, single-shot Held-Out Test Set evaluation, ROC/PR/Confusion Matrix plots, and serialization to `models/best_model_bundle.joblib`.
- **`notebooks/04_shap_stability_and_haddon_rq3.ipynb`**: Global and local TreeSHAP explanations, 5-fold rank stability evaluation (Spearman $\rho$ and Jaccard Top-10 overlap), and Haddon Matrix prevention table generation.
- **`app.py`**: Interactive three-tab Streamlit decision-support application providing live severity probability predictions, single-crash SHAP explanations, evaluation plots, and the Haddon Matrix guide.

---

## 3. Phase-by-Phase Process, Results & Observations

### 3.1 Data Understanding, Quality Audit & Preprocessing Preparation (`Notebook 01`)

#### 3.1.1 Dataset Integrity & Class Imbalance Profile
The raw Addis Ababa police crash dataset (`RTA Dataset.csv`) contains exactly **12,316 records** and **32 columns**, with **0 duplicate rows** and **0 missing values** in the target column (`Accident_severity`).

| Severity Class | Record Count | Percentage (%) | Numeric Code | Binary KSI Mapping |
| :--- | :---: | :---: | :---: | :--- |
| **Slight Injury** | 10,415 | 84.56% | `0` | Slight (`0`: 10,415 / 84.56%) |
| **Serious Injury** | 1,743 | 14.15% | `1` | Serious / Fatal KSI (`1`: 1,901 / 15.44%) |
| **Fatal injury** | 158 | 1.28% | `2` | Serious / Fatal KSI (`1`: 1,901 / 15.44%) |
| **Total** | **12,316** | **100.00%** | — | **12,316 (100.00%)** |

- **Observation**: The imbalance ratio between `Slight Injury` and `Fatal injury` is **`65.9 : 1`**. A trivial majority-class classifier predicting *Slight Injury* for every crash achieves **84.56% accuracy** while failing to detect a single fatal or serious crash (`Macro-F1 = 0.3054`). This empirically validates why accuracy was excluded as a model-selection criterion in favor of **Macro-F1**, **Serious/Fatal Recall**, and **Macro OvR ROC-AUC**.
- **Saved Figure**: `outputs/figures/01_target_distribution.png`.

#### 3.1.2 Post-Crash Data Leakage Exclusion
Seven attributes recorded *after* crash outcomes were permanently removed prior to analysis:
1. `Number_of_casualties`
2. `Casualty_class`
3. `Sex_of_casualty`
4. `Age_band_of_casualty`
5. `Casualty_severity`
6. `Work_of_casuality`
7. `Fitness_of_casuality`

Removing these 7 post-crash outcome columns left **24 legitimate pre-/at-crash predictors** known at the time of crash occurrence.

#### 3.1.3 String Cleaning & Encoding Repair
Raw categorical inspection uncovered systematic data-entry and character-encoding corruptions in `RTA Dataset.csv`, all of which were corrected programmatically in `clean_rta_data()`:
- **Leading/trailing whitespace**: Categories such as `'  Recreational areas'` (`327` rows) and `'Recreational areas'` (`1` row), `' Church areas'`, and `' Industrial areas'` were stripped so identical categories merged cleanly.
- **Character corruption (`?` instead of `-`)**: `'Lorry (41?100Q)'`, `'Lorry (11?40Q)'`, and `'Public (13?45 seats)'` were restored to `'Lorry (41-100Q)'`, `'Lorry (11-40Q)'`, and `'Public (13-45 seats)'`.
- **Substring replacement corruption**: In `Pedestrian_movement`, an upstream find-and-replace of `'na'` -> `'Not a Pedestrian'` in the public dataset had corrupted `'stationary'` into `'statioNot a Pedestrianry'` across `468` records (e.g., `'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle'`). All occurrences were restored to `'stationary'`.
- **Concatenated label repair**: `'Rural village areasOffice areas'` (`20` rows) in `Area_accident_occured` was cleaned to `'Rural village areas'`.
- **Case & token normalization**: Lowercase `'unknown'` (`33` rows in `Driving_experience`) and `'na'` tokens were standardized to `'Unknown'`, while undocumented numeric codes `'5'` (`36` rows) and `'7'` (`76` rows) in `Defect_of_vehicle` were labeled `'Defect code 5'` and `'Defect code 7'`.

#### 3.1.4 Missing-Value Profile & Imputation Strategy
Out of the 24 pre-crash predictors, 14 contained `NaN` values (`outputs/tables/01_missing_values_summary.csv`):

| Predictor | Null Count | Null (%) | Explicit `'Unknown'` Count | Combined Missing/Unknown (%) | Unique Categories |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Defect_of_vehicle` | 4,427 | 35.95% | 0 | 35.95% | 3 |
| `Service_year_of_vehicle` | 3,928 | 31.89% | 2,883 | 55.30% | 6 |
| `Type_of_vehicle` | 950 | 7.71% | 0 | 7.71% | 17 |
| `Types_of_Junction` | 887 | 7.20% | 191 | 8.75% | 8 |
| `Driving_experience` | 829 | 6.73% | 33 | 7.00% | 7 |
| `Educational_level` | 741 | 6.02% | 100 | 6.83% | 7 |
| `Vehicle_driver_relation` | 579 | 4.70% | 14 | 4.81% | 4 |
| `Owner_of_vehicle` | 482 | 3.91% | 0 | 3.91% | 4 |
| `Lanes_or_Medians` | 385 | 3.13% | 57 | 3.59% | 7 |
| `Vehicle_movement` | 308 | 2.50% | 88 | 3.22% | 13 |
| `Area_accident_occured` | 239 | 1.94% | 22 | 2.12% | 14 |
| `Road_surface_type` | 172 | 1.40% | 0 | 1.40% | 5 |
| `Type_of_collision` | 155 | 1.26% | 14 | 1.37% | 10 |
| `Road_allignment` | 142 | 1.15% | 0 | 1.15% | 9 |

*(All remaining 10 predictors—including `Time`, `Day_of_week`, `Age_band_of_driver`, `Sex_of_driver`, `Light_conditions`, `Weather_conditions`, `Road_surface_conditions`, `Number_of_vehicles_involved`, `Pedestrian_movement`, and `Cause_of_accident`—had `0.00%` nulls.)*

- **Decision & Justification**: Only two features (`Defect_of_vehicle` at `35.95%` and `Service_year_of_vehicle` at `31.89%`) exceeded the `20%` missingness threshold. Dropping incomplete rows would discard over `40%` of the dataset (losing scarce fatal cases), while mode-imputing `Defect_of_vehicle` into `'No defect'` would falsely label `4,427` uninspected vehicles as defect-free. Therefore, nulls in those two high-missingness columns were mapped to an explicit **`'Unknown'`** category, while low-missingness features (`1.15%–7.71%`) were imputed strictly **inside each training fold** (`most_frequent` for nominal, `-1` code for ordinal, `median` for numeric).
- **Saved Figure**: `outputs/figures/01_missing_values.png`.

#### 3.1.5 Multi-Vehicle Crash Event Grouping Discovery
Auditing timestamp sequences revealed a critical structural property of the Addis Ababa dataset:
- Across `12,316` rows, there are only **`2,755` unique `(Day_of_week, Time)` combinations**.
- **`6,221` records (`50.5%` of the entire dataset)** appear immediately adjacent to another record with the exact same `(Day_of_week, Time)` down to the minute, and **`97.69%` (`6,077 / 6,221`)** of those adjacent pairs share the exact same `Accident_severity`.
- **Implication**: Police crash logs recorded multiple rows (one per involved driver/vehicle) for the same multi-vehicle crash. Standard row-level random splitting places one vehicle from a crash into the Training set and another vehicle from the *same crash* into the Test set. Assigning `crash_event_id` via `(Day_of_week, Time)` ensured event-level independence during all downstream splits.

#### 3.1.6 Exploratory Data Analysis & Statistical Hypothesis Tests ($\chi^2$ and Cramér's V)
Visual and statistical exploration across `outputs/figures/01_temporal_heatmap.png`, `01_environmental_severity.png`, `01_circumstances_severity.png`, and `01_chi2_feature_ranking.png` established four major domain patterns:
1. **Temporal Volume vs. Temporal Severity Divergence**: Total crash counts peak sharply during weekday morning (`07:00–09:00`) and afternoon/evening (`16:00–19:00`) rush hours, whereas the **percentage of crashes resulting in Serious or Fatal injury rises markedly late at night and in the early morning (`20:00–05:00`)**, reflecting higher operating speeds, reduced visibility, and fatigue.
2. **Darkness Without Street Lighting**: Crashes under `'Darkness - no lighting'` exhibit a substantially higher combined Serious + Fatal rate (`~25%`) compared with `'Daylight'` (`~15%`) or `'Darkness - lights lit'` (`~16%`).
3. **Single-Vehicle & Pedestrian Vulnerability**: Crashes involving **`1` vehicle** (predominantly pedestrian knockdowns, rollovers, and fixed-object impacts) show nearly **double the Severe + Fatal rate (`~28.2%`: `519` Serious + `43` Fatal out of `1,996` crashes)** compared with 2-vehicle urban collisions (`~12.9%`: `963` Serious + `104` Fatal out of `8,340` crashes).
4. **Chi-Square ($\chi^2$) Independence & Cramér's V Ranking** (`outputs/tables/01_chi2_feature_relevance.csv`):
   - **8 out of 24 predictors** showed statistically significant individual associations with 3-class severity at $\alpha = 0.05$:
     1. `Number_of_vehicles_involved`: $\chi^2 = 317.55, \text{df} = 10, p = 3.01 \times 10^{-62}, V = 0.1135$
     2. `Hour`: $\chi^2 = 176.69, \text{df} = 46, p = 3.31 \times 10^{-17}, V = 0.0847$
     3. `Age_band_of_driver`: $\chi^2 = 60.53, \text{df} = 8, p = 3.67 \times 10^{-10}, V = 0.0496$
     4. `Types_of_Junction`: $\chi^2 = 58.94, \text{df} = 16, p = 7.88 \times 10^{-7}, V = 0.0489$
     5. `Day_of_week`: $\chi^2 = 47.20, \text{df} = 12, p = 4.30 \times 10^{-6}, V = 0.0438$
     6. `Light_conditions`: $\chi^2 = 45.02, \text{df} = 6, p = 4.63 \times 10^{-8}, V = 0.0428$
     7. `Area_accident_occured`: $\chi^2 = 42.75, \text{df} = 24, p = 0.0106, V = 0.0417$
     8. `Weather_conditions`: $\chi^2 = 41.75, \text{df} = 16, p = 4.30 \times 10^{-4}, V = 0.0412$
   - Because every individual Cramér's $V$ is `< 0.12`, crash severity cannot be explained by any single variable in isolation; rather, it emerges from **multi-variable non-linear interactions**, motivating ensemble and neural classifiers.

---

### 3.2 Objective 1: Leakage-Safe Splitting, Feature Engineering, Boruta & RQ1 (`Notebook 02`)

#### 3.2.1 Group-Aware Stratified 70 : 15 : 15 Split Verification
Using 20 non-overlapping folds of `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` grouped by `crash_event_id`, 14 folds (`70%`) were assigned to Training, 3 folds (`15%`) to Validation, and 3 folds (`15%`) to the final Held-Out Test set (`outputs/tables/02_split_verification.csv`):

| Split Partition | Total Rows | Share of Dataset (%) | Slight Count (%) | Serious Count (%) | Fatal Count (%) | Shared `crash_event_id` Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (70%)** | 8,620 | 69.99% | 7,329 (85.02%) | 1,172 (13.60%) | 119 (1.38%) | `0` |
| **Validation (15%)** | 1,867 | 15.16% | 1,552 (83.13%) | 291 (15.59%) | 24 (1.29%) | `0` |
| **Test (15%)** | 1,829 | 14.85% | 1,534 (83.87%) | 280 (15.31%) | 15 (0.82%) | `0` |
| **Total** | **12,316** | **100.00%** | **10,415 (84.56%)** | **1,743 (14.15%)** | **158 (1.28%)** | **Verified Disjoint** |

- **Saved Figure**: `outputs/figures/02_split_stratification_check.png`.

#### 3.2.2 Temporal Engineering, Encoding & Multicollinearity Check
Inside the pipeline, `TemporalFeatureExtractor` transforms `Time` and `Day_of_week` into four traffic-meaningful features and drops raw `Time` (`1,074` unique strings):
- `Hour` (`0–23` integer)
- `Time_of_day` (`Morning_Rush` [06:00–09:59], `Midday` [10:00–15:59], `Evening_Rush` [16:00–19:59], `Night` [20:00–05:59])
- `Is_rush_hour` (`1` if `06:00–09:59` or `16:00–19:59`, else `0`)
- `Is_weekend` (`1` if `Saturday` or `Sunday`, else `0`)

This produces **27 parent pre-crash features**:
- **4 Numerical/Binary features**: `Number_of_vehicles_involved`, `Hour`, `Is_rush_hour`, `Is_weekend` (median-imputed + `StandardScaler`).
- **5 Ordinal features**: `Age_band_of_driver`, `Educational_level`, `Driving_experience`, `Service_year_of_vehicle`, `Light_conditions` (mapped to domain-ordered integer scales in `OrdinalCategoryMapper`, `-1` for Unknown + `StandardScaler`).
- **18 Nominal features**: `Day_of_week`, `Sex_of_driver`, `Vehicle_driver_relation`, `Type_of_vehicle`, `Owner_of_vehicle`, `Defect_of_vehicle`, `Area_accident_occured`, `Lanes_or_Medians`, `Road_allignment`, `Types_of_Junction`, `Road_surface_type`, `Road_surface_conditions`, `Weather_conditions`, `Type_of_collision`, `Vehicle_movement`, `Pedestrian_movement`, `Cause_of_accident`, `Time_of_day` (mode-imputed + `OneHotEncoder(handle_unknown='ignore')` -> expanding to **157 total encoded columns**).

A Spearman rank-correlation audit across all 27 predictors on the Training set (`outputs/figures/02_feature_correlation_matrix.png`) confirmed **zero predictor pairs with $|r| > 0.75$**, ruling out severe multicollinearity.

#### 3.2.3 Boruta All-Relevant Feature Selection
Running `BorutaPy` (50 iterations, `perc=90`, balanced Random Forest estimator) strictly on the Training split (`outputs/tables/02_boruta_feature_selection.csv`, `outputs/figures/02_boruta_feature_ranking.png`) confirmed **8 core predictors** (`Rank = 1`) as statistically outperforming randomized shadow features:
1. `Hour` (`RF Importance = 0.1093`)
2. `Number_of_vehicles_involved` (`0.0752`)
3. `Day_of_week` (`0.0744`)
4. `Types_of_Junction` (`0.0735`)
5. `Age_band_of_driver` (`0.0585`)
6. `Is_weekend` (`0.0583`)
7. `Cause_of_accident` (`0.0503`)
8. `Time_of_day` (`0.0450`)

Secondary domain features (`Area_accident_occured` [Rank 2], `Type_of_vehicle` [Rank 3], `Vehicle_movement` [Rank 4], `Driving_experience` [Rank 5], `Lanes_or_Medians` [Rank 7], `Type_of_collision` [Rank 8], `Light_conditions` [Rank 10]) retained non-trivial tree split importance (`0.024–0.052`) and were kept in the full regularized pipeline so subtle multi-way interactions on rare fatal crashes were not prematurely discarded.

#### 3.2.4 Answering RQ1: 5-Fold Comparison of Imbalance-Handling Strategies
Three strategies—**No Balancing**, **Class Weighting**, and **SMOTE-NC** (applied strictly inside each training fold prior to one-hot encoding)—were evaluated across Decision Tree, Random Forest, and XGBoost using identical 5-fold `StratifiedGroupKFold` splits on the Training set (`outputs/tables/02_rq1_imbalance_comparison.csv`, `outputs/figures/02_rq1_imbalance_comparison.png`):

| Model | Imbalance Strategy | Macro-F1 (Mean ± SD) | Serious Recall (Mean ± SD) | Serious F1 | Fatal Recall (Mean ± SD) | Fatal F1 | Macro ROC-AUC | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | No Balancing | 0.3242 ± 0.0070 | 0.0380 ± 0.0146 | 0.0634 | 0.0000 ± 0.0000 | 0.0000 | 0.5335 | 0.8343 |
| **Decision Tree** | Class Weighting | 0.2696 ± 0.0426 | **0.4719 ± 0.0831** | 0.2292 | **0.3083 ± 0.1749** | 0.0404 | 0.5347 | 0.4081 |
| **Decision Tree** | SMOTE-NC | 0.3251 ± 0.0091 | 0.2823 ± 0.0320 | 0.1951 | 0.1317 ± 0.1301 | 0.0362 | 0.5207 | 0.6003 |
| **Random Forest** | No Balancing | 0.3063 ± 0.0031 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 ± 0.0000 | 0.0000 | 0.6096 | **0.8501** |
| **Random Forest** | Class Weighting | 0.3310 ± 0.0126 | 0.0576 ± 0.0163 | 0.0854 | 0.0071 ± 0.0160 | 0.0095 | 0.6025 | 0.8147 |
| **Random Forest** | SMOTE-NC | 0.3364 ± 0.0052 | 0.1280 ± 0.0170 | 0.1377 | 0.0071 ± 0.0160 | 0.0080 | 0.5565 | 0.7618 |
| **XGBoost** | No Balancing | 0.3125 ± 0.0023 | 0.0101 ± 0.0036 | 0.0195 | 0.0000 ± 0.0000 | 0.0000 | **0.6115** | 0.8485 |
| **XGBoost** | **Class Weighting** | **0.3553 ± 0.0229** | **0.3363 ± 0.0340** | **0.2320** | **0.1367 ± 0.0844** | **0.0562** | 0.5978 | 0.6470 |
| **XGBoost** | SMOTE-NC | 0.3517 ± 0.0187 | 0.1745 ± 0.0135 | 0.1640 | 0.0576 ± 0.0641 | 0.0475 | 0.5682 | 0.7330 |

- **Key RQ1 Observations**:
  1. **Failure of Unbalanced Training**: Without balancing, Random Forest (`85.01%` accuracy) and XGBoost (`84.85%` accuracy) achieved **`0.00%` Fatal Recall** and `0.00%–1.01%` Serious Recall—confirming that high accuracy on imbalanced crash data is an illusion caused by majority-class dominance.
  2. **Superiority of Cost-Sensitive Class Weighting**: Pairing **Class Weighting** with XGBoost produced the **highest Macro-F1 (`0.3553`)**, **highest Serious F1 (`0.2320`)**, and **highest Fatal F1 (`0.0562`)**, while maintaining higher ROC-AUC (`0.5978` vs. `0.5682` for SMOTE-NC). Furthermore, once Random Forest tree depth was properly regularized in Phase 4 (`max_depth=8`, `min_samples_leaf=8`, `balanced_subsample`), cost-sensitive Random Forest jumped to `0.3682` CV Macro-F1!
  3. **Why Class Weighting Outperformed SMOTE-NC on Categorical Crash Records**: Because 23 of the predictors are categorical, synthetic nearest-neighbor generation in `SMOTE-NC` across only ~95 fatal training cases per fold introduces noisy mode-voted category combinations that slightly degrade probability ranking (ROC-AUC dropped from `0.6096` to `0.5565` in Random Forest and `0.6115` to `0.5682` in XGBoost). Cost-sensitive weighting re-weights the loss on *authentic* historical crashes without inventing unrealistic combinations.

---

### 3.3 Objective 2: Model Tuning, Cross-Validation & Final Test Evaluation (`Notebook 03`)

#### 3.3.1 Hyperparameter Tuning & Overfitting Control
All four candidate models were tuned via randomized search over 5-fold `StratifiedGroupKFold` on the Training set (`8,620` rows), optimizing **Macro-F1** (`outputs/tables/03_hyperparameter_tuning_summary.csv`):

| Model | Role | Selected Optimal Hyperparameters | Train Macro-F1 | 5-Fold CV Macro-F1 | Generalization Gap |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Decision Tree** | Baseline | `criterion='gini', max_depth=14, min_samples_split=20, min_samples_leaf=5` | 0.4933 | 0.3088 | 0.1846 |
| **Random Forest** | Bagging Ensemble | `n_estimators=150, max_depth=8, min_samples_leaf=8, max_features='sqrt', class_weight='balanced_subsample'` | 0.6076 | **0.3682** | 0.2394 |
| **XGBoost** | Gradient Boosting | `n_estimators=150, learning_rate=0.02, max_depth=7, min_child_weight=5, subsample=0.7, colsample_bytree=0.7` | 0.6195 | 0.3584 | 0.2611 |
| **MLP** | Neural Network | `hidden_layer_sizes=(64,), activation='tanh', alpha=0.0001, learning_rate_init=0.001, early_stopping=True` | 0.9739 | 0.3544 | 0.6195 |

- **Observation**: Constraining tree depth (`max_depth=8`, `min_samples_leaf=8`) allowed `Random Forest` with `balanced_subsample` weights to prevent pure-leaf collapse and achieve the best tuning Macro-F1 (`0.3682`), whereas `MLPClassifier` (`157` sparse one-hot inputs + oversampled rare classes) memorized training patterns (`0.9739` Train F1) despite early stopping.

#### 3.3.2 Repeated Group-Stratified Cross-Validation Comparison (RQ2)
Evaluating the four tuned models across **10 folds (2 repeats × 5 group-stratified folds)** produced the core RQ2 benchmark (`outputs/tables/03_rq2_model_comparison_summary.csv`, `outputs/figures/03_rq2_cv_model_comparison.png`):

| Model | Macro-F1 (Mean ± SD) | Serious Recall (Mean ± SD) | Serious F1 (Mean ± SD) | Fatal Recall (Mean ± SD) | Fatal F1 (Mean ± SD) | Macro ROC-AUC (Mean ± SD) | Accuracy (Mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.3621 ± 0.0251** | 0.2701 ± 0.0247 | 0.2233 ± 0.0173 | 0.0961 ± 0.1340 | **0.0415 ± 0.0580** | 0.5602 ± 0.0659 | 0.7013 ± 0.0243 |
| **XGBoost** | 0.3552 ± 0.0155 | 0.3571 ± 0.0284 | **0.2440 ± 0.0164** | 0.0828 ± 0.0775 | 0.0373 ± 0.0313 | **0.5728 ± 0.0687** | 0.6561 ± 0.0289 |
| **MLP** | 0.3507 ± 0.0116 | 0.1709 ± 0.0199 | 0.1775 ± 0.0141 | 0.0073 ± 0.0160 | 0.0095 ± 0.0202 | 0.5500 ± 0.0442 | **0.7652 ± 0.0181** |
| **Decision Tree** | 0.3035 ± 0.0144 | **0.4089 ± 0.0648** | 0.2164 ± 0.0201 | **0.1112 ± 0.1004** | 0.0293 ± 0.0255 | 0.5212 ± 0.0272 | 0.5178 ± 0.0369 |

#### 3.3.3 Validation Check & Binary KSI Fallback (`Fatal + Serious` vs. `Slight`)
On the **15% Validation split (`1,867` crashes)** (`outputs/tables/03_validation_set_3class.csv`), **Random Forest** again ranked #1 (`Macro-F1 = 0.3608`), followed by **XGBoost** (`0.3531`), **MLP** (`0.3523`), and **Decision Tree** (`0.3035`), confirming **Random Forest** as the 3-class Champion (`Combined CV/Val Macro-F1 = 0.3614`).

Evaluating the **Section 3.2.4 Binary Fallback (`Fatal + Serious` KSI vs. `Slight`)** on the same Validation split (`outputs/tables/03_binary_ksi_comparison.csv`) yielded:

| Model | Binary Macro-F1 | Severe/Fatal Precision | Severe/Fatal Recall | Severe/Fatal F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.5603** | **0.2572** | 0.3397 | 0.2927 | 0.5697 | 0.2608 |
| **XGBoost** | 0.5469 | 0.2382 | 0.4000 | **0.2986** | 0.5671 | 0.2514 |
| **MLP** | 0.5285 | 0.2199 | 0.1968 | 0.2077 | 0.5494 | 0.2120 |
| **Decision Tree** | 0.4277 | 0.1752 | **0.5746** | 0.2685 | **0.5746** | **0.2641** |

- **Observation**: Merging the ultra-sparse `Fatal` class (`1.28%`) with `Serious` (`14.15%`) into the internationally standard **KSI (`15.44%`)** indicator raised Random Forest's Macro-F1 from `0.3608` to **`0.5603`** and PR-AUC to `0.2608` (well above the `0.1687` validation KSI prevalence), proving that extreme sample scarcity (`158` fatal rows) is the main limiting factor on 3-class fatal precision.

#### 3.3.4 Single-Shot Held-Out Test Set Evaluation (`1,829` Untouched Crashes)
Refitting all four tuned pipelines on the combined `85%` development set (`Train + Validation = 10,487` rows) and evaluating **once** on the untouched **15% Test Set (`1,829` rows: `1,534` Slight, `280` Serious, `15` Fatal)** yielded (`outputs/tables/03_final_test_set_results.csv`, `outputs/figures/03_test_confusion_matrices.png`, `03_test_roc_and_pr_curves.png`):

| Model | Test Macro-F1 | Slight Recall | Slight F1 | Serious Precision | Serious Recall | Serious F1 | Fatal Precision | Fatal Recall | Fatal F1 | Macro ROC-AUC | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.3844** | 0.7621 | 0.8127 | **0.2765** | 0.3357 | 0.3032 | **0.0205** | **0.2000** | **0.0373** | 0.5214 | 0.6922 |
| **XGBoost** | 0.3774 | 0.7151 | 0.7864 | 0.2596 | **0.4107** | **0.3181** | 0.0154 | 0.1333 | 0.0276 | **0.5694** | 0.6638 |
| **MLP** | 0.3499 | **0.8299** | **0.8397** | 0.2027 | 0.2179 | 0.2100 | 0.0000 | 0.0000 | 0.0000 | 0.5100 | **0.7294** |
| **Decision Tree (Baseline)** | 0.3082 | 0.4993 | 0.6318 | 0.1730 | 0.4893 | 0.2556 | **0.0205** | **0.2000** | **0.0373** | 0.5544 | 0.4954 |

- **Answer to RQ2**:
  - **Random Forest** achieved the best overall balance on the held-out Test Set (**Macro-F1 = `0.3844`**, beating the Decision Tree baseline `0.3082` by **`+7.62` percentage points**, or `+24.7%` relative gain), correctly identifying **`33.57%` (`94/280`)** of Serious crashes and **`20.00%` (`3/15`)** of Fatal crashes while maintaining **`76.21%`** recall on Slight crashes.
  - **XGBoost** finished a close second (**Macro-F1 = `0.3774`**) and achieved the **highest Serious F1 (`0.3181`)**, **highest Serious Recall among ensembles (`41.07%`)**, and **highest Macro ROC-AUC (`0.5694`)**.
  - **MLP** (`0.3499` Macro-F1) outperformed the Decision Tree in Macro-F1 but failed to detect any of the `15` test-set Fatal crashes (`0.00%` Fatal Recall), confirming that tree ensembles handle imbalanced, categorical police crash records more effectively than standard feedforward neural networks.

---

### 3.4 Objective 3: Explainable AI (SHAP), Cross-Fold Stability & Haddon Matrix (`Notebook 04`)

#### 3.4.1 Global & Local TreeSHAP Analysis
Using exact `shap.TreeExplainer` on the Champion **Random Forest** across all `1,829` Test records (`157` encoded columns aggregated additively back to the `27` parent attributes; `outputs/tables/04_global_shap_importance.csv`, `outputs/figures/04_shap_global_bar.png`, `04_shap_beeswarm_severe.png`):

| Rank | Crash Predictor | Mean `|SHAP|` (Slight) | Mean `|SHAP|` (Serious) | Mean `|SHAP|` (Fatal) | Mean `|SHAP|` (Severe/Fatal Avg) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `Is_weekend` | 0.0134 | 0.0158 | 0.0292 | **0.0225** |
| 2 | `Types_of_Junction` | 0.0142 | 0.0102 | 0.0242 | **0.0172** |
| 3 | `Time_of_day` | 0.0167 | 0.0087 | 0.0234 | **0.0161** |
| 4 | `Day_of_week` | 0.0091 | 0.0106 | 0.0190 | **0.0148** |
| 5 | `Number_of_vehicles_involved` | 0.0187 | 0.0127 | 0.0123 | **0.0125** |
| 6 | `Hour` | 0.0092 | 0.0066 | 0.0152 | **0.0109** |
| 7 | `Light_conditions` | 0.0055 | 0.0073 | 0.0128 | **0.0101** |
| 8 | `Lanes_or_Medians` | 0.0057 | 0.0061 | 0.0117 | **0.0089** |
| 9 | `Cause_of_accident` | 0.0064 | 0.0053 | 0.0113 | **0.0083** |
| 10 | `Defect_of_vehicle` | 0.0091 | 0.0037 | 0.0127 | **0.0082** |
| 11 | `Area_accident_occured` | 0.0055 | 0.0036 | 0.0086 | 0.0061 |
| 12 | `Age_band_of_driver` | 0.0063 | 0.0026 | 0.0086 | 0.0056 |
| 13 | `Driving_experience` | 0.0036 | 0.0037 | 0.0070 | 0.0054 |
| 14 | `Type_of_vehicle` | 0.0044 | 0.0034 | 0.0071 | 0.0052 |
| 15 | `Type_of_collision` | 0.0040 | 0.0031 | 0.0069 | 0.0050 |

- **Beeswarm & Local Waterfall Insights (`04_shap_beeswarm_severe.png`, `04_shap_local_cases.png`)**:
  - Fewer vehicles involved (`Number_of_vehicles_involved = 1`, representing pedestrian or single-vehicle rollover/fixed-object crashes), weekend travel (`Saturday`/`Sunday`), night-time hours (`Night` / unlit darkness), uncontrolled junction configurations (`Y Shape`, `Crossing`, `No junction` open stretches), undivided carriageways, and younger driver age bands (`18–30`, `Under 18`) consistently push SHAP contributions **positively** toward `Serious` and `Fatal` predictions.

#### 3.4.2 5-Fold Cross-Validation SHAP Stability Assessment (RQ3 Core Contribution)
Recomputing TreeSHAP rankings independently across all **5 group-stratified CV folds** on the Training set (`outputs/tables/04_shap_fold_stability.csv`, `outputs/figures/04_shap_fold_stability.png`) proved strong explanation stability:
- **Mean Pairwise Spearman Rank Correlation across all 10 fold pairs**: **$\rho = 0.7875$**
- **Mean Pairwise Top-10 Jaccard Similarity across all 10 fold pairs**: **$J = 0.7296$**

| Stability Order | Predictor | Mean Rank (± SD) | Top-10 Fold Frequency | Fold Ranks (F1–F5) | Certification Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `Types_of_Junction` | **1.00 ± 0.00** | **5 / 5** | 1, 1, 1, 1, 1 | **Stable ($\ge 4/5$ folds)** |
| **2** | `Is_weekend` | 3.00 ± 1.41 | **5 / 5** | 2, 5, 2, 4, 2 | **Stable ($\ge 4/5$ folds)** |
| **3** | `Day_of_week` | 4.20 ± 1.30 | **5 / 5** | 4, 6, 5, 3, 3 | **Stable ($\ge 4/5$ folds)** |
| **4** | `Number_of_vehicles_involved` | 4.40 ± 2.30 | **5 / 5** | 5, 8, 3, 2, 4 | **Stable ($\ge 4/5$ folds)** |
| **5** | `Time_of_day` | 4.80 ± 2.77 | **5 / 5** | 3, 2, 4, 6, 9 | **Stable ($\ge 4/5$ folds)** |
| **6** | `Hour` | 5.80 ± 1.64 | **5 / 5** | 7, 4, 8, 5, 5 | **Stable ($\ge 4/5$ folds)** |
| **7** | `Age_band_of_driver` | 8.20 ± 1.64 | **5 / 5** | 10, 10, 7, 7, 7 | **Stable ($\ge 4/5$ folds)** |
| **8** | `Cause_of_accident` | 9.20 ± 1.10 | **4 / 5** | 9, 9, 11, 9, 8 | **Stable ($\ge 4/5$ folds)** |
| **9** | `Lanes_or_Medians` | 9.40 ± 3.71 | **4 / 5** | 6, 15, 10, 10, 6 | **Stable ($\ge 4/5$ folds)** |
| *10* | *`Area_accident_occured`* | *10.80 ± 1.30* | *2 / 5* | *11, 12, 9, 12, 10* | *Secondary / Variable* |
| *11* | *`Road_surface_conditions`* | *14.60 ± 9.07* | *2 / 5* | *8, 3, 21, 16, 25* | *Secondary / Variable* |
| *12* | *`Light_conditions`* | *17.80 ± 7.16* | *1 / 5* | *14, 7, 22, 24, 22* | *Secondary / Variable* |

- **Critical Insight**: Notice how features like `Road_surface_conditions` (Rank 3 in Fold 2, but Rank 25 in Fold 5!) and `Light_conditions` (Rank 7 in Fold 2, Rank 24 in Fold 4) jumped dramatically depending on which rare crashes landed in a single fold. Without our 5-fold stability gate, a researcher analyzing Fold 2 alone would falsely report `Road_surface_conditions` as a Top-3 determinant! Filtering strictly to predictors appearing in the Top 10 in $\ge 4/5$ folds guarantees policy recommendations rest on reproducible associations.

#### 3.4.3 Haddon Matrix Road-Safety Prevention Synthesis
The 9 verified stable factors were mapped into the **Haddon Matrix** (`outputs/tables/04_haddon_matrix_recommendations.csv`), translating model explanations into Safe System countermeasures relevant to African urban corridors and Nigeria's **Federal Road Safety Corps (FRSC)**:

| Stable Factor (CV Consistency) | Haddon Dimension | Pre-Crash (Crash Prevention) | Crash Phase (Injury Mitigation) | Post-Crash (Life Preservation) |
| :--- | :--- | :--- | :--- | :--- |
| **`Types_of_Junction`** (5/5, Rank 1.0) | Road / Environment | Convert high-conflict X/Y crossings into roundabouts, channelized turns, or signalized intersections. | Install raised junction tables and speed humps to physically lower impact speeds. | Standardized junction signage/markers to accelerate emergency caller pinpointing. |
| **`Is_weekend`** (5/5, Rank 3.0) | Environment & Human | Targeted weekend speed & sobriety checkpoints near markets, worship centers, and leisure corridors. | Strict seat-belt, child-restraint, and motorcycle helmet enforcement on weekend trips. | Dedicated weekend trauma-team readiness and ambulance staging along arterial routes. |
| **`Day_of_week`** (5/5, Rank 4.2) | Environment & Human | Align patrol deployment schedules with high-severity Friday–Sunday travel peaks. | Consistent occupant restraint compliance across weekday commercial and weekend private traffic. | Trauma-center shift staffing calibrated to day-of-week severity patterns. |
| **`Number_of_vehicles_involved`** (5/5, Rank 4.4) | Road / Environment & Human | Segregated pedestrian walkways, speed calming in mixed-traffic zones, and roadside hazard clearance. | Forgiving roadside barriers, pedestrian-friendly vehicle fronts, and median guardrails. | Rapid emergency dispatch protocols along single-vehicle rollover/pedestrian corridors. |
| **`Time_of_day`** (5/5, Rank 4.8) | Environment & Human | Staggered commercial haulage scheduling and dusk-to-dawn speed enforcement. | Mandatory retro-reflective conspicuity tape on heavy trucks and buses operating after dusk. | Off-peak emergency hotline prioritization along inter-urban/arterial corridors. |
| **`Hour`** (5/5, Rank 5.8) | Environment & Human | Night-time speed, alcohol, and fatigue patrols (`20:00–05:00`) plus street-lighting upgrades. | High-visibility reflective road studs/markings and night-time restraint checks. | 24-hour trauma response readiness and dedicated night-shift ambulance staging. |
| **`Age_band_of_driver`** (5/5, Rank 8.2) | Human (Driver) | Graduated driver licensing, stricter novice probation rules, and youth safety campaigns. | Mandatory seat-belt and certified helmet enforcement among young drivers and riders. | Emergency medical record linkage via digital driver licence registries. |
| **`Cause_of_accident`** (4/5, Rank 9.2) | Human (Behavior) | Automated speed enforcement, overtaking/lane-discipline patrols, and defensive driver testing. | Commercial fleet speed-limiting devices (FRSC mandate) to cap impact kinetic energy. | Systematic post-crash blackspot audits linking police crash-cause logs to engineering fixes. |
| **`Lanes_or_Medians`** (4/5, Rank 9.4) | Road / Environment | Install physical median barriers or centerline rumble strips on undivided two-way corridors. | Energy-absorbing median guardrails preventing cross-median head-on intrusions. | Planned emergency turnaround gates integrated safely into divided medians. |

---

## 4. Mapping All Generated Artifacts to Your Thesis Document

Every figure (`outputs/figures/`) and CSV table (`outputs/tables/`) corresponds directly to a Chapter 4 table or figure in the updated Word document:

- **Section 4.2 (Data Audit & Cleaning)**: `01_target_distribution.png` (Fig 4.1), `01_missing_values.png` (Fig 4.2), `01_missing_values_summary.csv` (Table 4.1 & 4.2).
- **Section 4.3 (Exploratory & Statistical Analysis)**: `01_temporal_heatmap.png` (Fig 4.3), `01_environmental_severity.png` (Fig 4.4), `01_circumstances_severity.png` (Fig 4.5), `01_chi2_feature_ranking.png` (Fig 4.6), `01_chi2_feature_relevance.csv` (Table 4.3).
- **Section 4.4 (Splitting, Boruta & RQ1 Imbalance Results)**: `02_split_stratification_check.png` (Fig 4.7), `02_feature_correlation_matrix.png` (Fig 4.8), `02_boruta_feature_ranking.png` (Fig 4.9), `02_rq1_imbalance_comparison.png` (Fig 4.10), `02_split_verification.csv` (Table 4.4), `02_boruta_feature_selection.csv` (Table 4.5), `02_rq1_imbalance_comparison.csv` (Table 4.6).
- **Section 4.5 (Model Tuning, CV, Binary Fallback & Test Evaluation — RQ2)**: `03_rq2_cv_model_comparison.png` (Fig 4.11), `03_test_confusion_matrices.png` (Fig 4.12), `03_test_roc_and_pr_curves.png` (Fig 4.13), `03_hyperparameter_tuning_summary.csv` (Table 4.7), `03_rq2_model_comparison_summary.csv` (Table 4.8), `03_binary_ksi_comparison.csv` (Table 4.9), `03_final_test_set_results.csv` (Table 4.10).
- **Section 4.6 (SHAP Explainability, 5-Fold Stability & Haddon Matrix — RQ3)**: `04_shap_global_bar.png` (Fig 4.14), `04_shap_beeswarm_severe.png` (Fig 4.15), `04_shap_local_cases.png` (Fig 4.16), `04_shap_fold_stability.png` (Fig 4.17), `04_global_shap_importance.csv` (Table 4.11), `04_shap_fold_stability.csv` (Table 4.12), `04_haddon_matrix_recommendations.csv` (Table 4.13).
- **Section 4.7 (Interactive Decision-Support Prototype)**: `app.py` (Streamlit web application).
