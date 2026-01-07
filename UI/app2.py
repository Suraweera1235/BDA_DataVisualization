import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os

st.set_page_config(page_title="IMovie Managerial Dashboard", layout="wide")

# --- Load CSV safely ---
file_path = "data/Film_Dataset.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    st.error(f"CSV file not found! Please check the path: {file_path}")
    st.stop()

# --- Preprocessing ---
df['Release_Date'] = pd.to_datetime(df['Release_Date'])
df['Viewing_Month'] = pd.to_datetime(df['Viewing_Month'])
df['Release_Year'] = df['Release_Date'].dt.year
df['Release_Month'] = df['Release_Date'].dt.month
df['Viewing_Year'] = df['Viewing_Month'].dt.year
df['Month_Number'] = df['Viewing_Month'].dt.month
df['Movie_Age'] = 2025 - df['Release_Year']

# Keep original columns for display
df['Language_original'] = df['Language']
df['Category_original'] = df['Category']

# --- Sidebar Filters ---
st.sidebar.header("Filters")
languages = st.sidebar.multiselect("Select Language(s):", df['Language_original'].unique(), default=list(df['Language_original'].unique()))
categories = st.sidebar.multiselect("Select Category(s):", df['Category_original'].unique(), default=list(df['Category_original'].unique()))
years = st.sidebar.slider("Select Release Year Range:", int(df['Release_Year'].min()), int(df['Release_Year'].max()), (int(df['Release_Year'].min()), int(df['Release_Year'].max())))
months = st.sidebar.multiselect("Select Viewing Month(s):", list(range(1,13)), default=list(range(1,13)))

# --- Apply filters ---
filtered_df = df[
    df['Language_original'].isin(languages) &
    df['Category_original'].isin(categories) &
    df['Release_Year'].between(years[0], years[1]) &
    df['Month_Number'].isin(months)
].copy()

st.title("🎬 IMovie Managerial Dashboard")
st.markdown("Interactive dashboard for managerial insights: historical trends + predicted top movies.")

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust filters.")
    st.stop()

# --- KPI Metrics ---
total_views = filtered_df['Number_of_Views'].sum()
avg_rating = filtered_df['Viewer_Rate'].mean()
total_movies = filtered_df['Film_Name'].nunique()

top_language = filtered_df.groupby('Language_original')['Number_of_Views'].sum().idxmax()
top_category = filtered_df.groupby('Category_original')['Number_of_Views'].sum().idxmax()
highest_rated_movie = filtered_df.loc[filtered_df['Viewer_Rate'].idxmax()]['Film_Name']

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Views", f"{total_views:,}")
col2.metric("Average Rating", f"{avg_rating:.2f}")
col3.metric("Total Movies", f"{total_movies}")
col4.metric("Top Language", top_language)
col5.metric("Top Category", top_category)
col6.metric("Highest Rated Movie", highest_rated_movie)

st.markdown("---")

# --- Top 10 Historical Movies by Views ---
st.subheader("Top 10 Historical Movies by Views")
top_movies = filtered_df.groupby(['Film_Name','Language_original','Category_original'], as_index=False)['Number_of_Views'].sum()
top_movies = top_movies.sort_values('Number_of_Views', ascending=False).head(10)
st.dataframe(top_movies)

# --- Interactive Charts ---
st.subheader("Views by Category")
cat_chart = px.bar(filtered_df.groupby('Category_original', as_index=False)['Number_of_Views'].sum(),
                   x='Category_original', y='Number_of_Views', color='Category_original', title="Total Views by Category")
st.plotly_chart(cat_chart, use_container_width=True)

st.subheader("Views by Language")
lang_chart = px.bar(filtered_df.groupby('Language_original', as_index=False)['Number_of_Views'].sum(),
                    x='Language_original', y='Number_of_Views', color='Language_original', title="Total Views by Language")
st.plotly_chart(lang_chart, use_container_width=True)

st.subheader("Monthly View Trend")
monthly_trend = filtered_df.groupby('Viewing_Month', as_index=False)['Number_of_Views'].sum()
monthly_chart = px.line(monthly_trend, x='Viewing_Month', y='Number_of_Views', markers=True, title="Monthly Views Trend")
st.plotly_chart(monthly_chart, use_container_width=True)

st.subheader("Movie Age vs Number of Views")
scatter_chart = px.scatter(filtered_df, x='Movie_Age', y='Number_of_Views', color='Category_original',
                           hover_data=['Film_Name','Language_original'], size='Number_of_Views',
                           title="Movie Age vs Views by Category")
st.plotly_chart(scatter_chart, use_container_width=True)

st.markdown("---")

# --- Search / Select Specific Movies ---
st.subheader("Search or Select Specific Movies")
movie_select = st.multiselect("Select Movies to Compare:", filtered_df['Film_Name'].unique())
if movie_select:
    selected_movies = filtered_df[filtered_df['Film_Name'].isin(movie_select)]
    st.dataframe(selected_movies)

# --- Predictive Model for Selected Month ---
st.subheader("Predicted Top Movies")
month_select = st.selectbox("Select Month for Prediction", list(range(1,13)), index=11)  # Default December

# One-hot encoding for model
df_model = df.copy()
df_model = pd.get_dummies(df_model, columns=['Category_original','Language_original'], drop_first=True)

# Features and target
X_cols = df_model.drop(['Number_of_Views','Film_Name','Viewing_Month','Release_Date'], axis=1).columns
X = df_model[X_cols]
y = df_model['Number_of_Views']

# Chronological split
n_total = len(df_model)
n_train = int(n_total * 0.8)
X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]
X_test  = X.iloc[n_train:]
y_test  = y.iloc[n_train:]

# Train RandomForest
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

# Predict for selected month
pred_month_data = df[df['Month_Number']==month_select].copy()
pred_month_model = pd.get_dummies(pred_month_data, columns=['Category_original','Language_original'], drop_first=True)

# Ensure all columns exist
for col in X_cols:
    if col not in pred_month_model.columns:
        pred_month_model[col] = 0

pred_month_model = pred_month_model[X_cols]  # reorder columns
pred_month_data['Predicted_Views'] = rf_model.predict(pred_month_model)

top_pred = pred_month_data[['Film_Name','Release_Date','Predicted_Views']].sort_values('Predicted_Views', ascending=False).head(10)
st.dataframe(top_pred.style.format({"Predicted_Views": "{:,.0f}"}))

# Predicted views chart
pred_chart = px.bar(top_pred, x='Film_Name', y='Predicted_Views', hover_data=['Release_Date'],
                    title=f"Top Predicted Movies for Month {month_select}")
st.plotly_chart(pred_chart, use_container_width=True)

# --- Download Filtered Data ---
# -----------------------------

st.download_button(
    label="Download Top December 2025 Movies",
    data=top_december.to_csv(index=False),
    file_name="Top_December_2025_Movies.csv",
    mime="text/csv"
)


st.markdown("---")

# --- Key Takeaways ---
st.subheader("Key Takeaways")
st.markdown("""
- Romance movies in Japanese, Chinese, and French languages show **highest predicted views** for December 2025.
- Older movies with high viewer ratings continue to attract significant views — consider **re-releases or bundled promotions**.
- Monthly trend analysis helps plan **targeted marketing campaigns** in months with historically higher engagement.
- Managers can filter by language, category, and release year to focus on **specific market segments**.
""")
