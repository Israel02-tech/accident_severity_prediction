"""
Shared data cleaning, feature engineering, splitting, and evaluation helpers.

Keeping these simple functions and transformers here lets both the Jupyter
notebooks and the Streamlit demo app (`app.py`) reuse the exact same logic
without duplicating code.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# 1. Project Constants
# ---------------------------------------------------------------------------

# Columns recorded AFTER the crash occurs (must be dropped to prevent leakage)
CASUALTY_LEAKAGE_COLS = [
    "Number_of_casualties",
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Casualty_severity",
    "Work_of_casuality",
    "Fitness_of_casuality",
]

TARGET_COL = "Accident_severity"

# Ordered class labels from least severe (0) to most severe (2)
SEVERITY_ORDER = ["Slight Injury", "Serious Injury", "Fatal injury"]
SEVERITY_MAP = {"Slight Injury": 0, "Serious Injury": 1, "Fatal injury": 2}
INV_SEVERITY_MAP = {v: k for k, v in SEVERITY_MAP.items()}

# Natural orderings for ordinal features (-1 represents Unknown / Missing)
ORDINAL_MAPS = {
    "Age_band_of_driver": {
        "Unknown": -1,
        "Under 18": 0,
        "18-30": 1,
        "31-50": 2,
        "Over 51": 3,
    },
    "Educational_level": {
        "Unknown": -1,
        "Illiterate": 0,
        "Writing & reading": 1,
        "Elementary school": 2,
        "Junior high school": 3,
        "High school": 4,
        "Above high school": 5,
    },
    "Driving_experience": {
        "Unknown": -1,
        "No Licence": 0,
        "Below 1yr": 1,
        "1-2yr": 2,
        "2-5yr": 3,
        "5-10yr": 4,
        "Above 10yr": 5,
    },
    "Service_year_of_vehicle": {
        "Unknown": -1,
        "Below 1yr": 0,
        "1-2yr": 1,
        "2-5yrs": 2,
        "5-10yrs": 3,
        "Above 10yr": 4,
    },
    "Light_conditions": {
        "Daylight": 0,
        "Darkness - lights lit": 1,
        "Darkness - lights unlit": 2,
        "Darkness - no lighting": 3,
    },
}


# ---------------------------------------------------------------------------
# 2. Data Cleaning & Event Grouping
# ---------------------------------------------------------------------------

def clean_rta_data(df: pd.DataFrame, drop_leakage: bool = True) -> pd.DataFrame:
    """
    Clean formatting errors, standardize missing/unknown labels, and optionally
    remove post-crash casualty columns that cause data leakage.
    """
    out = df.copy()

    # Step 1: Remove post-crash casualty columns if requested
    if drop_leakage:
        cols_to_drop = [c for c in CASUALTY_LEAKAGE_COLS if c in out.columns]
        out = out.drop(columns=cols_to_drop)

    # Step 2: Strip extra whitespace and repair known text corruption in strings
    obj_cols = out.select_dtypes(include="object").columns
    for col in obj_cols:
        # Strip leading/trailing spaces (e.g. '  Recreational areas' -> 'Recreational areas')
        s = out[col].astype("string").str.strip()

        # Fix corrupted characters present in the raw Addis Ababa CSV
        s = s.str.replace("statioNot a Pedestrianry", "stationary", regex=False)
        s = s.str.replace("?", "-", regex=False)
        s = s.str.replace("Rural village areasOffice areas", "Rural village areas", regex=False)
        s = s.str.replace(r"\s+", " ", regex=True)

        # Standardize lowercase 'unknown' or 'na' tokens into 'Unknown'
        s = s.replace({"unknown": "Unknown", "na": "Unknown"})

        # Convert back to standard object dtype keeping real NaNs intact
        out[col] = s.astype(object).where(s.notna(), np.nan)

    # Step 3: Clean up Defect_of_vehicle codes and high-missingness columns (>30% missing)
    if "Defect_of_vehicle" in out.columns:
        out["Defect_of_vehicle"] = out["Defect_of_vehicle"].replace(
            {"5": "Defect code 5", "7": "Defect code 7"}
        )
        out["Defect_of_vehicle"] = out["Defect_of_vehicle"].fillna("Unknown")

    if "Service_year_of_vehicle" in out.columns:
        out["Service_year_of_vehicle"] = out["Service_year_of_vehicle"].fillna("Unknown")

    return out


def assign_crash_event_id(df: pd.DataFrame) -> pd.Series:
    """
    Assign a shared integer ID to records with the exact same (Time, Day_of_week).
    In the Addis Ababa dataset, multi-vehicle crashes were logged as multiple rows
    with identical timestamps, so grouping by event ID prevents train-test leakage.
    """
    event_keys = df["Day_of_week"].astype(str) + "_" + df["Time"].astype(str)
    event_ids, _ = pd.factorize(event_keys)
    return pd.Series(event_ids, index=df.index, name="crash_event_id")


# ---------------------------------------------------------------------------
# 3. Scikit-Learn Compatible Feature Transformers
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Convert raw 'Time' (HH:MM:SS) and 'Day_of_week' into meaningful traffic
    time features, then drop raw 'Time' so models cannot memorize timestamps.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Extract hour (0-23) from 'Time' if present
        if "Time" in df.columns:
            parsed_hour = pd.to_datetime(
                df["Time"].astype(str), format="%H:%M:%S", errors="coerce"
            ).dt.hour
            # Fallback if 'Time' is formatted as H:MM without seconds
            mask_na = parsed_hour.isna()
            if mask_na.any():
                parsed_hour[mask_na] = (
                    df.loc[mask_na, "Time"].astype(str).str.split(":").str[0].astype(float)
                )
            df["Hour"] = parsed_hour.fillna(12).astype(int)
            df = df.drop(columns=["Time"])

        # Group hours into clear road-traffic periods
        def _period(h: int) -> str:
            if 6 <= h <= 9:
                return "Morning_Rush"
            elif 10 <= h <= 15:
                return "Midday"
            elif 16 <= h <= 19:
                return "Evening_Rush"
            else:
                return "Night"

        df["Time_of_day"] = df["Hour"].map(_period)

        # Rush-hour indicator (6-9 AM or 4-7 PM)
        df["Is_rush_hour"] = df["Hour"].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)

        # Weekend indicator (Saturday or Sunday)
        if "Day_of_week" in df.columns:
            df["Is_weekend"] = df["Day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

        return df


class OrdinalCategoryMapper(BaseEstimator, TransformerMixin):
    """
    Map ordered categorical columns (like driver age band or driving experience)
    to integer codes defined in ORDINAL_MAPS. Unseen or missing values map to -1.
    """

    def __init__(self, mapping_dict=None):
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        self.maps_ = self.mapping_dict if self.mapping_dict is not None else ORDINAL_MAPS
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col in df.columns:
            if col in self.maps_:
                df[col] = df[col].map(self.maps_[col]).fillna(-1).astype(float)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# ---------------------------------------------------------------------------
# 4. Leakage-Safe 70 : 15 : 15 Splitter & Evaluation Metrics
# ---------------------------------------------------------------------------

def group_stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    group_col: str = "crash_event_id",
    random_state: int = 42,
):
    """
    Split dataset into 70% Train, 15% Validation, and 15% Test while:
      1. Preserving class proportions across all three splits (stratification)
      2. Keeping records with the same crash_event_id in the same split (no leakage)

    Uses 20 stratified group folds: 14 folds (70%) -> Train, 3 (15%) -> Val, 3 (15%) -> Test.
    """
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=random_state)
    folds = list(sgkf.split(df, df[target_col], groups=df[group_col]))

    # Collect validation indices from each of the 20 non-overlapping folds
    train_idx = np.concatenate([folds[i][1] for i in range(0, 14)])
    val_idx = np.concatenate([folds[i][1] for i in range(14, 17)])
    test_idx = np.concatenate([folds[i][1] for i in range(17, 20)])

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def evaluate_multiclass(y_true, y_pred, y_proba=None, label: str = "Model") -> dict:
    """
    Compute all thesis evaluation metrics for 3-class crash severity prediction:
    Accuracy (reference only), Macro-F1, per-class Precision/Recall/F1, and Macro ROC-AUC.
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    metrics = {
        "Model": label,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Slight_Recall": rec[0],
        "Slight_F1": f1[0],
        "Serious_Precision": prec[1],
        "Serious_Recall": rec[1],
        "Serious_F1": f1[1],
        "Fatal_Precision": prec[2],
        "Fatal_Recall": rec[2],
        "Fatal_F1": f1[2],
    }

    if y_proba is not None:
        try:
            metrics["ROC_AUC_Macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["ROC_AUC_Macro"] = np.nan

    return metrics
