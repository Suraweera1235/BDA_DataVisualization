# BDA Dashboard

This Streamlit dashboard displays EDA plots and predictions derived from `Models/BDA.ipynb` logic.

Running locally

1. Create a Python environment and install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run the Streamlit app from the repository root:

```powershell
streamlit run UI/app.py
```

Notes

- The app attempts to locate `Film_Dataset.csv` in the repository root and in `data/`.
- Use the sidebar to retrain the Random Forest and adjust `n_estimators` and number of top films shown for December.
