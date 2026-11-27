# Road Traffic Accident Severity Prediction & Explainable AI

B.Sc. Computer Science Project — **University of Ibadan**  
*Development of an Artificial Intelligence / Machine Learning Model for Identifying Patterns in Road Traffic Accidents*

---

## Project Structure

```text
accident_severity_model_prediction/
├── RTA Dataset.csv                                    # Raw Addis Ababa crash dataset (12,316 records)
├── Progress_Comfort_Project_Concise (2).docx          # Thesis Chapters 1–3
├── requirements.txt                                   # Pinned Python dependencies
├── src/
│   └── utils.py                                       # Shared cleaning, time-feature, split & metric helpers
├── data/
│   └── processed/                                     # Cleaned dataset + fixed 70:15:15 splits
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb                 # Data quality audit, leakage removal, Chi-Square & EDA
│   ├── 02_preprocessing_and_imbalance_rq1.ipynb       # Group-safe splitting, Boruta & RQ1 imbalance study
│   ├── 03_model_training_and_evaluation_rq2.ipynb     # Tuning & evaluating DT, RF, XGBoost, MLP (RQ2)
│   └── 04_shap_stability_and_haddon_rq3.ipynb         # Global/local SHAP, 5-fold stability & Haddon Matrix (RQ3)
├── models/
│   └── best_model_bundle.joblib                       # Saved champion model & metadata for Streamlit demo
├── outputs/
│   ├── figures/                                       # High-resolution (300 DPI) figures for Chapter 4
│   └── tables/                                        # CSV summary tables for Chapter 4
└── app.py                                             # Interactive Streamlit demo with live SHAP explanations
```

## Quick Start

1. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Open the Jupyter notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```

3. **Launch the interactive Streamlit demonstration app**:
   ```bash
   streamlit run app.py
   ```
