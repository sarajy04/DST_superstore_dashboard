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
    
    #Selection box for analysis type
    analysis_type = st.selectbox("View Sales Based On :", ["Timeline", "Category", "Geographical Location", "All of the Above"])
    
    # Visualization 1 - Line graph for sales over time
    if analysis_type == "Timeline":
        st.subheader("Line Graph for Sales Over Time")

        # Time granularity selection
        time_granularity = st.selectbox("Select Time Granularity:", ["Monthly", "Yearly"])

        # Group data by Monthly or Yearly
        if time_granularity == "Monthly":
            sales_over_time = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
            sales_over_time['Order Date'] = sales_over_time['Order Date'].dt.to_timestamp()
        else:  
            sales_over_time = df.groupby(df['Order Date'].dt.to_period('Y'))['Sales'].sum().reset_index()
            sales_over_time['Order Date'] = sales_over_time['Order Date'].dt.to_timestamp()

        # Plot line graph
        fig = px.line(sales_over_time, x='Order Date', y='Sales', 
                      title=f"{time_granularity} Sales Trend", 
                      labels={'Order Date': 'Date', 'Sales': 'Sales (USD)'})
        st.plotly_chart(fig)
    
    #category visualization 
    elif analysis_type == "Category":
        st.subheader("Sales Distribution per Category")
        # Group by category and sum sales, then sort
        Top_category = df.groupby("Category")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)

        # Find total revenue generated across all categories
        total_revenue_category = Top_category["Sales"].sum()

        # Convert the total revenue to an integer, then string, then add '$' sign
        total_revenue_category = f"${int(total_revenue_category)}"
        # pie chart for top 3 cat
        plt.rcParams["figure.figsize"] = (13,5) # width and height of figure is defined in inches
        plt.rcParams['font.size'] = 12.0 # Font size is defined
        plt.rcParams['font.weight'] = 6 # Font weight is defined
        # we don't want to look at the percentage distribution in the pie chart. Instead, we want to look at the exact revenue generated by the categories.
        def autopct_format(values): 
            def my_format(pct): 
                total = sum(values) 
                val = int(round(pct*total/100.0))
                return ' ${v:d}'.format(v=val)
            return my_format
        colors = ['#BC243C','#FE840E','#C62168'] # Colors are defined for the pie chart
        explode = (0.05,0.05,0.05)
        fig1, ax1 = plt.subplots()
        ax1.pie(Top_category['Sales'], colors = colors, labels=Top_category['Category'], autopct= autopct_format(Top_category['Sales']), startangle=90,explode=explode)
        centre_circle = plt.Circle((0,0),0.82,fc='white') # drawing a circle on the pie chart to make it look better 
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle) # Add the circle on the pie chart
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal') 
        # we can look the total revenue generated by all the categories at the center
        label = ax1.annotate('Total Sales \n'+str(total_revenue_category),color = 'red', xy=(0, 0), fontsize=12, ha="center")
        plt.tight_layout()
        plt.show()
        st.pyplot(fig) 

        # category selection
        category_selection = st.selectbox("Select :", ["Category", "Sub-Category"])

        # Group data by Category
        category_sales = df.groupby('Category')['Sales'].sum().reset_index()

        # Plot bar chart
        fig = px.bar(category_sales, x='Category', y='Sales', 
                     title="Category-wise Sales Distribution", 
                     labels={'Category': 'Category', 'Sales': 'Sales (USD)'})
        st.plotly_chart(fig)

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
