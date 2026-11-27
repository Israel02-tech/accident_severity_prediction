"""
Interactive Crash Severity Prediction & SHAP Explainability Demo (Section 3.6.4)

Run locally with:
    .venv/bin/streamlit run app.py
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT_ROOT / "models" / "best_model_bundle.joblib"
HADDON_PATH = PROJECT_ROOT / "outputs" / "tables" / "04_haddon_matrix_recommendations.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


@st.cache_resource
def load_bundle():
    """Load the saved model bundle once and cache it across user interactions."""
    return joblib.load(BUNDLE_PATH)


def compute_single_row_parent_shap(explain_pipe, input_df: pd.DataFrame, target_class_idx: int):
    """
    Compute exact TreeSHAP values for a single crash scenario and sum one-hot
    encoded contributions back to their human-readable parent features.
    """
    prep = explain_pipe.named_steps["prep"]
    clf = explain_pipe.named_steps["clf"]

    parent_df = prep.named_steps["temporal"].transform(input_df)
    parent_features = list(parent_df.columns)

    X_enc = prep.transform(input_df)
    enc_names = list(prep.named_steps["encode"].get_feature_names_out())

    explainer = shap.TreeExplainer(clf)
    raw_shap = explainer.shap_values(X_enc)
    shap_3d = np.stack(raw_shap, axis=-1) if isinstance(raw_shap, list) else np.asarray(raw_shap)

    parent_contribs = np.zeros(len(parent_features))
    for j, ec in enumerate(enc_names):
        matched_pf = ec
        for pf in sorted(parent_features, key=len, reverse=True):
            if ec == pf or ec.startswith(pf + "_"):
                matched_pf = pf
                break
        parent_contribs[parent_features.index(matched_pf)] += shap_3d[0, j, target_class_idx]

    out = pd.DataFrame({
        "Feature": [f"{f} = {parent_df.iloc[0][f]}" for f in parent_features],
        "SHAP_Contribution": parent_contribs,
        "Abs_SHAP": np.abs(parent_contribs),
    }).sort_values("Abs_SHAP", ascending=True).tail(10)
    return out


def main():
    st.set_page_config(
        page_title="Road Traffic Crash Severity — ML & SHAP Demo",
        page_icon="🚦",
        layout="wide",
    )

    st.title("🚦 Road Traffic Accident Severity — Explainable ML Decision Support")
    st.caption(
        "B.Sc. Computer Science Project Demonstration • Predicts crash severity from pre-/at-crash "
        "conditions and explains key risk factors using SHAP (Decision support only — associations, not causation)."
    )

    if not BUNDLE_PATH.exists():
        st.error("Model bundle not found! Please run Notebooks 01–04 first to generate `models/best_model_bundle.joblib`.")
        st.stop()

    bundle = load_bundle()
    pipe = bundle["pipeline"]
    tree_pipe = bundle["best_tree_pipeline"]
    raw_cols = bundle["raw_feature_cols"]
    ui_options = bundle["ui_options"]
    defaults = bundle["default_values"]
    class_names = bundle["class_names"]
    test_metrics = bundle["test_metrics"]

    # Sidebar: Key Crash Scenario Inputs
    st.sidebar.header("🛠️ Configure Crash Conditions")
    st.sidebar.caption("Adjust the main scenario fields below (remaining fields default to dataset typical values).")

    key_fields = [
        ("Time", "Crash Hour"),
        ("Day_of_week", "Day of Week"),
        ("Light_conditions", "Lighting Condition"),
        ("Weather_conditions", "Weather Condition"),
        ("Road_surface_conditions", "Road Surface Condition"),
        ("Types_of_Junction", "Junction Type"),
        ("Lanes_or_Medians", "Lanes / Median Layout"),
        ("Area_accident_occured", "Area of Crash"),
        ("Type_of_collision", "Collision Type"),
        ("Number_of_vehicles_involved", "Number of Vehicles Involved"),
        ("Type_of_vehicle", "Primary Vehicle Type"),
        ("Vehicle_movement", "Vehicle Movement"),
        ("Cause_of_accident", "Recorded Cause / Maneuver"),
        ("Age_band_of_driver", "Driver Age Band"),
        ("Driving_experience", "Driver Experience"),
    ]

    user_inputs = dict(defaults)
    for col, friendly_label in key_fields:
        opts = ui_options[col]
        default_idx = opts.index(defaults[col]) if defaults[col] in opts else 0
        user_inputs[col] = st.sidebar.selectbox(friendly_label, options=opts, index=default_idx)

    with st.sidebar.expander("Additional Driver / Road Fields"):
        shown_cols = {k for k, _ in key_fields}
        for col in raw_cols:
            if col not in shown_cols:
                opts = ui_options[col]
                default_idx = opts.index(defaults[col]) if defaults[col] in opts else 0
                user_inputs[col] = st.selectbox(col.replace("_", " "), options=opts, index=default_idx)

    input_df = pd.DataFrame([user_inputs])[raw_cols]

    tab_pred, tab_charts, tab_haddon = st.tabs([
        "🔍 Live Prediction & SHAP Explanation",
        "📊 Model Evaluation & Stability Charts",
        "🛡️ Haddon Matrix Prevention Guide",
    ])

    with tab_pred:
        probs = pipe.predict_proba(input_df)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        severe_or_fatal_prob = float(probs[1] + probs[2])

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Crash Severity", pred_label)
        col_b.metric("Combined Serious / Fatal Risk", f"{severe_or_fatal_prob * 100:.1f}%")
        col_c.metric(
            f"Champion Model ({bundle['model_name']})",
            f"Test Macro-F1: {test_metrics['Macro_F1']:.3f}",
            f"ROC-AUC: {test_metrics.get('ROC_AUC_Macro', 0):.3f}",
        )

        left_col, right_col = st.columns([1, 1.35])

        with left_col:
            st.subheader("Predicted Class Probabilities")
            fig_p, ax_p = plt.subplots(figsize=(5.5, 3.8))
            colors_p = ["#4C72B0", "#DD8452", "#C44E52"]
            bars = ax_p.bar(class_names, probs * 100, color=colors_p, edgecolor="black")
            ax_p.set_ylabel("Predicted Probability (%)")
            ax_p.set_ylim(0, 108)
            for b, p_val in zip(bars, probs * 100):
                ax_p.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{p_val:.1f}%", ha="center", fontweight="bold")
            st.pyplot(fig_p)
            plt.close(fig_p)

        with right_col:
            explain_cls_idx = pred_idx if pred_idx > 0 else 1
            st.subheader(f"Top Factors Driving '{class_names[explain_cls_idx]}' Risk (SHAP)")
            shap_df = compute_single_row_parent_shap(tree_pipe, input_df, target_class_idx=explain_cls_idx)

            fig_s, ax_s = plt.subplots(figsize=(7.2, 4.2))
            bar_c = np.where(shap_df["SHAP_Contribution"] >= 0, "#C44E52", "#4C72B0")
            ax_s.barh(shap_df["Feature"], shap_df["SHAP_Contribution"], color=bar_c, edgecolor="black")
            ax_s.axvline(0, color="black", linewidth=1)
            ax_s.set_xlabel("SHAP Contribution (Red = Increases Risk, Blue = Decreases Risk)")
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)

    with tab_charts:
        st.subheader("Key Experimental Results (Notebooks 02–04)")
        c1, c2 = st.columns(2)
        if (FIG_DIR / "03_rq2_cv_model_comparison.png").exists():
            c1.image(str(FIG_DIR / "03_rq2_cv_model_comparison.png"), caption="RQ2: 10-Fold Cross-Validation Comparison")
        if (FIG_DIR / "03_test_confusion_matrices.png").exists():
            c2.image(str(FIG_DIR / "03_test_confusion_matrices.png"), caption="Held-Out Test Set Confusion Matrices")
        c3, c4 = st.columns(2)
        if (FIG_DIR / "04_shap_global_bar.png").exists():
            c3.image(str(FIG_DIR / "04_shap_global_bar.png"), caption="RQ3: Global SHAP Importance")
        if (FIG_DIR / "04_shap_fold_stability.png").exists():
            c4.image(str(FIG_DIR / "04_shap_fold_stability.png"), caption="RQ3: 5-Fold SHAP Rank Stability")

    with tab_haddon:
        st.subheader("Evidence-Based Prevention Measures (Haddon Matrix)")
        st.markdown(
            "Only crash predictors verified as **stable across 5-fold cross-validation** in Notebook 04 "
            "are linked below to Pre-Crash, Crash, and Post-Crash interventions."
        )
        if HADDON_PATH.exists():
            st.dataframe(pd.read_csv(HADDON_PATH), use_container_width=True, hide_index=True)
        else:
            st.info("Run Notebook 04 to populate `outputs/tables/04_haddon_matrix_recommendations.csv`.")


if __name__ == "__main__":
    main()
