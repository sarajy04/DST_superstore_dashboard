import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Superstore Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Load data with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv')
        
        # data cleaning
        # Drop 'Row ID' column
        df.drop('Row ID', axis=1, inplace=True)
        
        # Convert date columns to datetime format
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
        
        # Create 'Month_order' and 'Year_order' columns
        df['Month_order'] = df['Order Date'].dt.to_period('M')
        df['Year_order'] = df['Order Date'].dt.to_period('Y')
        
        # Fill missing values in 'Postal Code'
        df['Postal Code'] = df['Postal Code'].fillna(5401)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Check if data is loaded
if df.empty:
    st.warning("No data available. Please check:")
    st.markdown("""
    - train.csv exists in the current directory
    - File contains 'Order Date' and 'Ship Date' columns
    - You're using the correct Kaggle dataset
    """)
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Visualizations", "Predictive Model"])

# Introduction page
if page == "Introduction":
    st.title("Introduction of Superstore Dashboard🛍️")
    st.image("shopping.jpg", use_container_width=True)
    st.markdown("""
    ## Overview and Aims
    This dashboard aims to provide a comprehensive analysis and forecasting capabilities for sales data,empowering data-driven decision-making for optimizing business strategies
    Including:
    - Historical sales trends
    - Product performance tracking
    - Regional sales breakdown
    - Machine learning-based sales forecasting
    """)
    
    st.subheader("📋 Dataset Overview")
    st.write(df.head())
    st.write(f"Total records: {len(df)}")
    st.write(f"Total variables: {len(df.columns)}")
    st.write(f"Time range: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
   
    # Create a summary DataFrame
    st.write(f"**Dataset Features and Their Data Types:**")

    info_df = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': [df[col].dtype for col in df.columns],
    })
    
    # Rearranging the columns for better readability
    info_df = info_df[['Column Name', 'Data Type']]
    
    st.write(info_df)

# Visualizations page
elif page == "Visualizations":
    st.title("📊 Sales Performance Analysis")
    
    # Date range filter
    date_options = df['Order Date'].dt.date.unique()
    start_date = st.sidebar.date_input("Start Date", min(date_options))
    end_date = st.sidebar.date_input("End Date", max(date_options))
    
    # Other filters
    with st.sidebar:
        categories = st.multiselect("Product Category", df['Category'].unique(), default=df['Category'].unique())
        regions = st.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
        segments = st.multiselect("Customer Segment", df['Segment'].unique(), default=df['Segment'].unique())

    filtered_df = df[
        (df['Order Date'].dt.date.between(start_date, end_date)) &
        (df['Category'].isin(categories)) &
        (df['Region'].isin(regions)) &
        (df['Segment'].isin(segments))
    ]

    # KPIs
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
    with col2:
        st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
    with col3:
        st.metric("Average Order Value", f"${filtered_df['Sales'].mean():,.2f}")
    with col4:
        st.metric("Unique Customers", filtered_df['Customer ID'].nunique())

    # Time series
    st.header("Sales Trends")
    monthly_sales = filtered_df.resample('M', on='Order Date')['Sales'].sum().reset_index()
    fig = px.line(monthly_sales, x='Order Date', y='Sales', title='Monthly Sales Trend')
    st.plotly_chart(fig, use_container_width=True)

    # Category and region breakdown
    st.header("Sales Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig = px.pie(category_sales, names='Category', values='Sales', title='Category Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
        fig = px.bar(region_sales, x='Region', y='Sales', title='Regional Performance')
        st.plotly_chart(fig, use_container_width=True)

    # Product performance
    st.header("Top Products")
    top_products = filtered_df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
    fig = px.bar(top_products, x='Product Name', y='Sales', title='Top 10 Products')
    st.plotly_chart(fig, use_container_width=True)

# Predictive Model page
else:
    st.title("🔮 Sales Forecasting")
    st.markdown("Use historical data to predict future sales")

    # Prepare data for Prophet
    @st.cache_data
    def prepare_forecast_data(df):
        daily_sales = df.resample('D', on='Order Date')['Sales'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        return daily_sales

    forecast_data = prepare_forecast_data(df)

    # Model parameters
    with st.sidebar:
        periods = st.slider("Forecast Period (days)", 30, 365, 90)
        seasonality = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
        changepoint = st.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05, 0.01)

    # Train model
    @st.cache_resource
    def train_model(data):
        model = Prophet(
            seasonality_mode=seasonality,
            changepoint_prior_scale=changepoint
        )
        model.fit(data)
        return model

    model = train_model(forecast_data)

    # Generate forecast
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Show results
    st.header("Forecast Results")
    fig = plot_plotly(model, forecast, xlabel="Date", ylabel="Sales")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Forecast Components"):
        fig2 = model.plot_components(forecast)
        st.write(fig2)

    st.markdown("### Forecast Statistics")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

# Footer
st.sidebar.markdown("---")
