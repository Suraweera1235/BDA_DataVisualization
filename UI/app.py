import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error


# ------------------------- Configuration / Constants -------------------------
APP_TITLE = "IMovie — December 2025 Marketing Strategy"
PRIMARY_COLOR = "#0b6efd"   # single professional blue (kept minimal)
ACCENT_COLOR = "#0f1724"    # dark accent
DATA_PATH = os.path.join("data", "Film_Dataset.csv")
MODEL_CACHE_PATH = "imovie_rf_model.joblib"
TEAM_REG_NUMBERS = ["1", "2", "3"]  # <-- replace these

# ------------------------- Page config & style -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

def local_css(theme="light"):
    if theme == "dark":
        bg = "#0b0f16"
        fg = "#e6eef8"
        card = "#0f1724"
        subtext = "#ffffff"
        button_text = "#000000" 
          # <-- black text for buttons
    else:
        bg = "#ffffff"
        fg = "#0f1724"
        card = "#f6f8fb"
        subtext = "#000000"
        button_text = "#ffffff"

    st.markdown(f"""
    <style>
    /* general app styling */
    .reportview-container {{background: {bg}; color: {fg};}}
    .stApp {{background: {bg}; color: {fg};}}
    .card {{background: {card}; padding: 12px; border-radius: 8px;}}
    .heading {{color: {PRIMARY_COLOR};}}
    .small-muted {{color: rgba(255,255,255,0.6); font-size:12px}}

    /* Form labels, placeholders, and input labels */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {{
        color: {subtext} !important;
    }}

    /* Metric subtext */
    .stMetric .stMetricDelta, .stMetric .stMetricValue {{
        color: {subtext} !important;
    }}

    /* Plotly axis titles and tick labels */
    .main .plotly .xtick text, .main .plotly .ytick text, .main .plotly .axis-title {{
        fill: {subtext} !important;
    }}

    /* Streamlit button text color (including form submit buttons) */
    div.stButton > button, form.stForm div.stButton > button {{
        color: {button_text} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ------------------------- Utilities: Preprocessing -------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess(path=DATA_PATH):
    df = pd.read_csv(path)

    # basic cleaning & datetime conversion (apply same pipeline as training)
    df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
    df['Viewing_Month'] = pd.to_datetime(df['Viewing_Month'], errors='coerce')

    # drop rows with invalid dates (if any)
    df = df.dropna(subset=['Release_Date', 'Viewing_Month']).reset_index(drop=True)

    # create features
    df['Release_Year'] = df['Release_Date'].dt.year
    df['Release_Month'] = df['Release_Date'].dt.month
    df['Viewing_Year'] = df['Viewing_Month'].dt.year
    df['Movie_Age'] = 2025 - df['Release_Year']
    df['Month_Number'] = df['Viewing_Month'].dt.month

    # keep originals for display
    df['Language_original'] = df.get('Language', np.nan)
    df['Category_original'] = df.get('Category', np.nan)

    # separate out December 2025 onward to avoid leakage (same as training code)
    dec_2025_onwards = df[df['Viewing_Month'] >= '2025-12-01']

    # keep training data before December 2025
    df_train_ready = df[df['Viewing_Month'] < '2025-12-01'].copy()

    # encode categorical variables (one-hot; drop_first=True)
    df_model = pd.get_dummies(df_train_ready, columns=['Category', 'Language'], drop_first=True)

    # sort chronologically
    df_model = df_model.sort_values('Viewing_Month').reset_index(drop=True)

    return df_model, dec_2025_onwards

# ------------------------- Modeling Utilities -------------------------
@st.cache_resource
def train_model(df_model):
    # time-based split (80/20)
    n_total = len(df_model)
    n_train = int(n_total * 0.8)
    train_data = df_model.iloc[:n_train]
    test_data = df_model.iloc[n_train:]

    drop_cols = ['Number_of_Views', 'Film_Name', 'Viewing_Month', 'Release_Date','Language_original','Category_original']
    X_train = train_data.drop([c for c in drop_cols if c in train_data.columns], axis=1)
    y_train = train_data['Number_of_Views']

    X_test = test_data.drop([c for c in drop_cols if c in test_data.columns], axis=1)
    y_test = test_data['Number_of_Views'] if 'Number_of_Views' in test_data.columns else None

    # baseline model (RandomForest)
    base = RandomForestRegressor(n_estimators=300, random_state=42)

    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(
        base,
        param_grid,
        n_iter=10,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_

    # evaluate on test set
    preds = best.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # save model
    joblib.dump({'model': best, 'feature_columns': X_train.columns.tolist()}, MODEL_CACHE_PATH)

    return {
        'model': best,
        'feature_columns': X_train.columns.tolist(),
        'metrics': {'r2': r2, 'mae': mae, 'rmse': rmse},
        'train_index': (0, n_train),
        'test_index': (n_train, n_total),
        'X_test': X_test,
        'y_test': y_test
    }

# load model from disk if exists (fast)
@st.cache_resource
def load_cached_model():
    if os.path.exists(MODEL_CACHE_PATH):
        cached = joblib.load(MODEL_CACHE_PATH)

        # ensure full dictionary format
        if 'metrics' not in cached:
            return None   # force retrain

        return cached
    return None

# predict helper
def predict_single(model_dict, input_df):
    model = model_dict['model']
    cols = model_dict['feature_columns']
    X = input_df.reindex(columns=cols, fill_value=0)
    return model.predict(X)

# ------------------------- App Layout -------------------------
col1, col2 = st.columns([8, 2])

with col1:
    st.markdown(f"<h1 class='heading'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.write("Managerial summary: Use the charts below to guide targeted December 2025 marketing spend.")

with col2:
    theme = st.toggle("Dark Mode", value=True)

# Dynamic toggle label color (WORKS)
st.markdown(
    f"""
    <style>
    [data-testid="stWidgetLabel"] * {{
        color: {"white" if theme else "black"} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Apply theme CSS
local_css('dark' if theme else 'light')



# Load and preprocess data
with st.spinner("Loading & preprocessing data..."):
    df_model, dec_future = load_and_preprocess(DATA_PATH)



# # show top KPI row
# c1, c2, c3, c4 = st.columns([2,2,2,2])

# c1.markdown(f"<div style='color:#0d47a1; font-size:28px; font-weight:bold;'>32</div><div>Films (unique)</div>", unsafe_allow_html=True)
# c2.markdown(f"<div style='color:#1565c0; font-size:28px; font-weight:bold;'>12</div><div>Languages (one-hot)</div>", unsafe_allow_html=True)
# c3.markdown(f"<div style='color:#1976d2; font-size:28px; font-weight:bold;'>6</div><div>Categories (one-hot)</div>", unsafe_allow_html=True)
# c4.markdown(f"<div style='color:#1e88e5; font-size:28px; font-weight:bold;'>443</div><div>Rows (training)</div>", unsafe_allow_html=True)


# total_films = len(df_model['Film_Name'].unique()) if 'Film_Name' in df_model.columns else df_model.shape[0]
# unique_langs = len([c for c in df_model.columns if c.startswith('Language_')])
# unique_cats = len([c for c in df_model.columns if c.startswith('Category_')])


# st.markdown("---")

# calculate KPIs
views_2025 = df_model[df_model['Viewing_Month'].dt.year == 2025]
total_views_2025 = views_2025['Number_of_Views'].sum()
top_genre = views_2025.groupby('Category_original')['Number_of_Views'].sum().idxmax()
most_viewed_movie = views_2025.loc[views_2025['Number_of_Views'].idxmax(), 'Film_Name']
highest_rated_movie = views_2025.loc[views_2025['Viewer_Rate'].idxmax(), 'Film_Name'] if 'Viewer_Rate' in views_2025.columns else "N/A"
# show KPI row
c1, c2, c3, c4 = st.columns([2,2,2,2])

c1.markdown(f"<div style='color:#0d47a1; font-size:28px; font-weight:bold;'>{total_views_2025:,}</div><div>Total Views in 2025</div>", unsafe_allow_html=True)
c2.markdown(f"<div style='color:#1565c0; font-size:28px; font-weight:bold;'>{top_genre}</div><div>Top Genre</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='color:#1976d2; font-size:28px; font-weight:bold;'>{most_viewed_movie}</div><div>Most-viewed Movie</div>", unsafe_allow_html=True)
c4.markdown(f"<div style='color:#1e88e5; font-size:28px; font-weight:bold;'>{highest_rated_movie}</div><div>Highest-rated Movie</div>", unsafe_allow_html=True)

st.markdown("---")


# Train or load model
model_info = load_cached_model()
if model_info is None:
    
    with st.spinner("Training RandomForest (RandomizedSearchCV)..."):
        model_info = train_model(df_model)
    #st.success("Model trained and cached.")
else:
    st.info("Loaded cached model.")

# Show model metrics
# metrics = model_info['metrics']
# st.subheader("Model performance (time-split test set)")
# st.write(f"R²: {metrics['r2']:.3f} — MAE: {metrics['mae']:.1f} — RMSE: {metrics['rmse']:.1f}")

# ------------------------- Managerial Plots -------------------------
st.subheader("Managerial Visualisations")

orig = pd.read_csv(DATA_PATH)
orig['Release_Date'] = pd.to_datetime(orig['Release_Date'], errors='coerce')
orig['Viewing_Month'] = pd.to_datetime(orig['Viewing_Month'], errors='coerce')
orig = orig[orig['Viewing_Month'] < '2025-12-01']

blue_colors = ['#0d47a1', '#1976d2', '#42a5f5', '#90caf9', '#64b5f6', '#1e88e5']

# Pie: Language distribution (default Plotly colors)
lang_counts = orig['Language'].fillna('Unknown').value_counts().reset_index()
lang_counts.columns = ['Language', 'Count']
fig_lang = px.pie(
    lang_counts,
    names='Language',
    values='Count',
    title='Language Distribution (Historical)'
)
fig_lang.update_layout(title_x=0.5)


# Pie: Category distribution (default Plotly colors)
cat_counts = orig['Category'].fillna('Unknown').value_counts().reset_index()
cat_counts.columns = ['Category', 'Count']
fig_cat = px.pie(
    cat_counts,
    names='Category',
    values='Count',
    title='Category Distribution (Historical)'
)
fig_cat.update_layout(title_x=0.5)

# Histogram: Release year (shades of blue)
fig_release = px.histogram(
    orig,
    x='Release_Date',
    nbins=20,
    title='Releases Over Time'
)
fig_release.update_traces(marker_color='#1976d2')
fig_release.update_layout(
    title_x=0.5,
    title_font=dict(color='white' if theme=='Dark' else 'black'),
    bargap=0.1,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Bar chart: Avg views (shades of blue)
if 'Number_of_Views' in orig.columns:
    avg_by_cat = orig.groupby('Category')['Number_of_Views'].mean().reset_index().sort_values(
        'Number_of_Views', ascending=False
    )
    fig_avg_cat = px.bar(
        avg_by_cat,
        x='Category',
        y='Number_of_Views',
        title='Average Views by Category',
        color='Category',
        color_discrete_sequence=blue_colors
    )
    fig_avg_cat.update_layout(
        title_x=0.5,
        title_font=dict(color='white' if theme=='Dark' else 'black'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
else:
    fig_avg_cat = go.Figure()

# Row 1
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.plotly_chart(fig_lang, use_container_width=True)
with row1_col2:
    st.plotly_chart(fig_cat, use_container_width=True)

# Row 2
row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.plotly_chart(fig_release, use_container_width=True)
with row2_col2:
    st.plotly_chart(fig_avg_cat, use_container_width=True)



st.subheader("Monthly Viewing Trend")

monthly = df_model.groupby('Viewing_Month')["Number_of_Views"].sum().reset_index()
fig = px.line(monthly, x="Viewing_Month", y="Number_of_Views", markers=True)
st.plotly_chart(fig)


# st.subheader("Rating vs Views Scatter Plot")

# fig = px.scatter(df_model, x="Viewer_Rate", y="Number_of_Views", color="Category_original", size="Number_of_Views")
# st.plotly_chart(fig)

# st.subheader("New Releases Impact")

# fig = px.box(df_model, x="Release_Year", y="Number_of_Views")
# st.plotly_chart(fig)

    

# ------------------------- Predictions for December (managerial) -------------------------
st.subheader("Top predicted films for December (managerial focus)")

# load feature columns
feature_cols = model_info['feature_columns']
model = model_info['model']

# prepare december dataset from the full CSV (including December rows) and apply same feature engineering
full = pd.read_csv(DATA_PATH)
full['Release_Date'] = pd.to_datetime(full['Release_Date'], errors='coerce')
full['Viewing_Month'] = pd.to_datetime(full['Viewing_Month'], errors='coerce')
full['Release_Year'] = full['Release_Date'].dt.year
full['Release_Month'] = full['Release_Date'].dt.month
full['Viewing_Year'] = full['Viewing_Month'].dt.year
full['Movie_Age'] = 2025 - full['Release_Year']
full['Month_Number'] = full['Viewing_Month'].dt.month
full['Language_original'] = full.get('Language')
full['Category_original'] = full.get('Category')

# Create dummy variables on full (but we will only predict for December rows)
full_model = pd.get_dummies(full, columns=['Category', 'Language'], drop_first=True)

# select December rows
december_rows = full_model[full_model['Month_Number'] == 12].copy()

if december_rows.empty:
    st.warning("No December rows found in dataset.")
else:
    # ensure prediction columns align with features
    X_dec = december_rows.reindex(columns=feature_cols, fill_value=0)
    december_rows['Predicted_Views'] = model.predict(X_dec)

    top_dec = december_rows.sort_values('Predicted_Views', ascending=False).head(10)

    display_cols = ['Film_Name', 'Language_original', 'Category_original', 'Release_Date', 'Predicted_Views']
    present = top_dec[display_cols].copy()
    present['Predicted_Views'] = present['Predicted_Views'].round(0).astype(int)

    # ---- RENAME COLUMNS ----
    present = present.rename(columns={
        'Language_original': 'Language',
        'Category_original': 'Category'
    })

    present['Release_Date'] = present['Release_Date'].dt.date

    # reset index
    present = present.reset_index(drop=True)

    # shift index to start from 1
    present.index = present.index + 1

    st.write("Top 10 films (predicted views) for December — use these as targets for promotions:")
    st.dataframe(present)
    

    # bar chart of top predictions
    present['Film_Language'] = present['Film_Name'] + " (" + present['Language'] + ")"
    fig_top = px.bar(
        present,
        x='Film_Language',
        y='Predicted_Views',
        title='Top 10 Predicted Views (December)',
        color='Film_Language',
        color_discrete_sequence=['#0d47a1', '#1565c0', '#1976d2', '#1e88e5', '#2196f3',
                                 '#42a5f5', '#64b5f6', '#90caf9', '#bbdefb', '#82b1ff']
    )

    y_min = present['Predicted_Views'].min() * 0.95
    y_max = present['Predicted_Views'].max() * 1.05
    fig_top.update_yaxes(range=[y_min, y_max])

    # show values on top of bars
    fig_top.update_traces(text=present['Predicted_Views'], textposition='outside')

    fig_top.update_layout(
        title_x=0.5,
        title_font=dict(color='white' if theme=='Dark' else 'black'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ------------------------- Feature Importance -------------------------
# st.subheader("Feature importance (managerial interpretation)")
# try:
#     importances = model.feature_importances_
#     fi = (
#         pd.DataFrame({'feature': feature_cols, 'importance': importances})
#         .sort_values('importance', ascending=False)
#         .head(15)
#     )

#     fig_fi = px.bar(
#         fi,
#         x='importance',
#         y='feature',
#         orientation='h',
#         title='Top 15 Feature Importances',
#         color='feature',
#         color_discrete_sequence=['#0d47a1', '#1565c0', '#1976d2', '#1e88e5', '#2196f3',
#                                  '#42a5f5', '#64b5f6', '#90caf9', '#bbdefb']
#     )
#     fig_fi.update_layout(
#         title_x=0.5,
#         title_font=dict(color='white' if theme=='Dark' else 'black'),
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         showlegend=False
#     )

#     st.plotly_chart(fig_fi, use_container_width=True)

# except Exception as e:
#     st.write("Could not compute feature importances:", e)

# ------------------------- Predict using user inputs -------------------------
# ------------------------- Predict using user inputs -------------------------
# st.subheader("Predict views for a custom film / scenario")
# with st.form(key='predict_form'):
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         film_name = st.text_input('Film name', value='My Test Film')
#         release_date = st.date_input('Release date', value=datetime(2023,1,1))
#         language = st.selectbox('Language', options=sorted(orig['Language'].dropna().unique()))
#     with col2:
#         category = st.selectbox('Category', options=sorted(orig['Category'].dropna().unique()))
#         viewing_month = st.date_input('Viewing month (choose any date within the month)', value=datetime(2025,12,1))
#         movie_age = 2025 - release_date.year
#     with col3:
#         # additional numeric features if present
#         extra_info = st.number_input('Optional: Known past average views per similar film (leave 0 if unknown)', min_value=0)

#     # Submit button
#     submitted = st.form_submit_button('Predict views')

# if submitted:
#     # build a single-row dataframe aligned with feature columns
#     row = {
#         'Film_Name': film_name,
#         'Release_Date': pd.to_datetime(release_date),
#         'Viewing_Month': pd.to_datetime(viewing_month),
#         'Release_Year': release_date.year,
#         'Release_Month': release_date.month,
#         'Viewing_Year': pd.to_datetime(viewing_month).year,
#         'Movie_Age': movie_age,
#         'Month_Number': pd.to_datetime(viewing_month).month,
#         'Language': language,
#         'Category': category
#     }
#     # create a one-row DataFrame and dummify
#     single = pd.DataFrame([row])
#     single_model = pd.get_dummies(single, columns=['Category', 'Language'], drop_first=True)

#     # align with feature columns
#     single_X = single_model.reindex(columns=feature_cols, fill_value=0)

#     pred = model.predict(single_X)[0]
#     st.success(f"Predicted Number_of_Views: {int(round(pred))}")
#     st.info("Use managerial plots and this prediction to decide where to allocate marketing budget (top languages/categories, and top predicted films).")


    # ------------------------- Appendix / Methods -------------------------
st.markdown("---")
st.header("Appendix: Methods & Notes")
st.markdown(
"""
**Data preprocessing**: Converted `Release_Date` and `Viewing_Month` to datetime, extracted year/month features, computed `Movie_Age` as 2025 - Release_Year, filtered out rows from Dec 1, 2025 onwards during training to avoid leakage, one-hot encoded `Category` and `Language`.

**Modeling**: Time-based 80/20 split (chronological). `RandomForestRegressor` tuned with `RandomizedSearchCV`. Performance measured using R², MAE, RMSE on the held-out time-split test set.

**Managerial visualisations**: Language and Category pie charts for distribution, release timeline histogram, average views by category, top predicted films for December, and feature importance. These charts focus on high-level insights for marketing decisions.
"""
)

