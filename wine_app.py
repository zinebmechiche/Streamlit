import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_wine


wine_data = load_wine()

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
df['target'] = wine_data['target']

# Sidebar Header
st.sidebar.markdown("<h2 style='color: #FF4B4B;'>Filter Parameters</h2>", unsafe_allow_html=True)

st.markdown("<h1 style='color: #6c757d; text-align: center;'>🍷 Wine Dataset Explorer 🍷</h1>", unsafe_allow_html=True)

# Select target (wine class)
selected_class = st.sidebar.multiselect(
    'Select Wine Class (Target)',
    df['target'].unique(),
    default=df['target'].unique() 
)

# Slider for Alcohol Content Range
alcohol_range = st.sidebar.slider(
    'Alcohol Content Range (%)',
    min_value=float(df['alcohol'].min()), 
    max_value=float(df['alcohol'].max()), 
    value=(df['alcohol'].min(), df['alcohol'].max()),
    step=0.1
)

# Slider for Malic Acid Range
malic_acid_range = st.sidebar.slider(
    'Malic Acid Range',
    min_value=float(df['malic_acid'].min()), 
    max_value=float(df['malic_acid'].max()), 
    value=(df['malic_acid'].min(), df['malic_acid'].max()),
    step=0.1
)

# Filter the DataFrame based on user inputs
filtered_df = df[
    (df['target'].isin(selected_class)) &
    (df['alcohol'] >= alcohol_range[0]) &
    (df['alcohol'] <= alcohol_range[1]) &
    (df['malic_acid'] >= malic_acid_range[0]) &
    (df['malic_acid'] <= malic_acid_range[1])
]


col1, col2 = st.columns(2)

# Scatter plot: Alcohol vs Malic Acid
with col1:
    st.subheader('Alcohol vs Malic Acid')
    fig = px.scatter(
        filtered_df, x='alcohol', y='malic_acid', color='target',
        title="Alcohol vs Malic Acid by Wine Class",
        labels={'alcohol': 'Alcohol (%)', 'malic_acid': 'Malic Acid'},
        hover_name='target', template="plotly_dark",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')))
    st.plotly_chart(fig)

# 3D Scatter plot for Alcohol, Malic Acid, and Ash
with col2:
    st.subheader('3D Scatter Plot (Alcohol, Malic Acid, Ash)')
    fig_3d = px.scatter_3d(
        filtered_df, x='alcohol', y='malic_acid', z='ash', color='target',
        title="3D Scatter Plot (Alcohol, Malic Acid, Ash)",
        labels={'alcohol': 'Alcohol (%)', 'malic_acid': 'Malic Acid', 'ash': 'Ash'},
        hover_name='target', template="plotly_dark",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_3d.update_traces(marker=dict(size=5, opacity=0.9))
    st.plotly_chart(fig_3d)


# Histogram of Alcohol Content
st.markdown("<h3 style='color: #FFC300;'>Alcohol Content Distribution by Wine Class</h3>", unsafe_allow_html=True)
fig_hist = px.histogram(
    filtered_df, x='alcohol', color='target', 
    title='Alcohol Content Distribution', nbins=20,
    labels={'target': 'Wine Class'},
    template="plotly_dark"
)
fig_hist.update_layout(bargap=0.2)
st.plotly_chart(fig_hist)

# Histogram of Malic Acid Distribution
st.markdown("<h3 style='color: #FF5733;'>Malic Acid Distribution by Wine Class</h3>", unsafe_allow_html=True)
fig_hist_acidity = px.histogram(
    filtered_df, x='malic_acid', color='target', 
    title='Malic Acid Distribution', nbins=20,
    labels={'target': 'Wine Class'},
    template="plotly_dark"
)
fig_hist_acidity.update_layout(bargap=0.2)
st.plotly_chart(fig_hist_acidity)

st.markdown("<h3 style='color: #3498db;'>Filtered Wine Data</h3>", unsafe_allow_html=True)
st.dataframe(filtered_df.style.background_gradient(cmap='Blues'))
