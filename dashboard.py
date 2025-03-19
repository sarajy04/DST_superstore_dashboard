import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
    
    # Selection box for analysis type
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

        st.markdown(f"""
        ### {time_granularity} Sales Trend Analysis
        - **Purpose**: 
            - Analyze sales trends over time to identify patterns, seasonality, and fluctuations.
        - **Granularity Options**:
            - Monthly: Aggregates sales data by month for detailed short-term trends.
            - Yearly: Aggregates sales data by year for long-term performance insights.
        - **Visualization**:
            - Line graph displaying sales trends over the selected time granularity.
        - **Insights**:
            - Highlights periods of high or low sales performance.
            - Helps identify seasonal trends and growth opportunities.
        - **Actionable Use**:
            - Inform strategic decisions for inventory management, marketing campaigns, and resource allocation.
        """)
    
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
        
    # Visualization 3 - Sales by Geographical Location
    elif analysis_type == "Geographical Location":
        st.subheader("Sales by States")
        
        # Define state and state_code lists
        state = ['Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 
                'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 
                'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 
                'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
                'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
                'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

        state_code = ['AL','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
                    'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD',
                    'TN','TX','UT','VT','VA','WA','WV','WI','WY']

        # Create a DataFrame for states and their codes
        state_df = pd.DataFrame({'State Code': state_code, 'State': state})

        # Ensure numeric columns are summed
        numeric_columns = df.select_dtypes(include=['number']).columns
        sales = df.groupby("State")[numeric_columns].sum().sort_values("Sales", ascending=False)

        # Reset index to make "State" a column
        sales.reset_index(inplace=True)

        # Drop "Postal Code" safely using axis=1
        if 'Postal Code' in sales.columns:
            sales.drop('Postal Code', axis=1, inplace=True)

        # Sort by "State"
        sales = sales.sort_values('State', ascending=True).reset_index(drop=True)

        # Merge sales with state codes
        sales = sales.merge(state_df, on="State", how="left")

        # Add text labels
        sales['text'] = sales['State'] + '<br>Sales: ' + sales['Sales'].astype(str)

        # Create the Choropleth map
        fig = go.Figure(data=go.Choropleth(
            locations=sales['State Code'],  # Spatial coordinates
            text=sales['text'],
            z=sales['Sales'].astype(float),  # Data to be color-coded
            locationmode='USA-states',  # Set of locations match entries in `locations`
            colorscale='Blues',
            colorbar_title="Sales",
        ))

        fig.update_layout(
            geo_scope='usa',  # Limit map scope to USA
        )

        # Display the map 
        st.plotly_chart(fig)

        st.markdown("""
        ### Geographical Sales Analysis
        - **Purpose**: 
            - Visualize sales distribution across different states in the USA.
        - **Visualization**: 
            - Interactive choropleth map with color-coded sales data by state.
            - Hover over states to view detailed sales figures.
        - **Insights**: 
            - Identify regions with high or low sales performance.
            - Highlight geographical trends and disparities in sales.
        - **Actionable Use**: 
            - Optimize regional strategies for marketing, inventory, and resource allocation.
            - Focus efforts on underperforming regions or capitalize on high-performing areas.
        """)

        # Sales Trend Analysis by State
        st.subheader("Sales Trend Analysis by State")
        
        # Get unique states from the dataset
        states = df['State'].unique()

        # State selection dropdown
        selected_state = st.selectbox("Select a state to view sales trend:", states)

        # Filter data for the selected state
        state_data = df[df['State'] == selected_state]

        # Group data by month and calculate sales for the selected state
        state_sales_trend = state_data.groupby(state_data['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        state_sales_trend['Order Date'] = state_sales_trend['Order Date'].dt.to_timestamp()

        # Plot line graph for sales trend by state
        fig = px.line(state_sales_trend, x='Order Date', y='Sales',
                title=f"Sales Trend for {selected_state} State",
                labels={'Order Date': 'Date', 'Sales': 'Sales (USD)'},
                template="plotly_white")
        st.plotly_chart(fig)

        st.markdown(""" 
        - **Purpose**: 
            - Analyze sales trends for the selected state over time.
        - **Visualization**: 
            -Line graph showing monthly sales trends for the selected state.
        - **Insights**:
            - Identify periods of high or low sales performance in the state.
            - Understand seasonal trends and growth opportunities specific to the state.
        - **Actionable Use**:
            - Tailor marketing and sales strategies to the state's performance trends.
            - Optimize inventory and resource allocation for the state.
        """)
        
        # Overall sales trend by region
        subheader = st.subheader("Overall Sales Trend by Region")
        
        # Multi-select dropdown for state filtering
        selected_states = st.multiselect("Select States to View Trends:", options=df['State'].unique(), default=df['State'].unique())

        # Filter data based on selected states
        filtered_data = df[df['State'].isin(selected_states)]

        # Group data by State and Month, then calculate total sales
        state_sales_trend = filtered_data.groupby([filtered_data['Order Date'].dt.to_period('M'), 'State'])['Sales'].sum().reset_index()
        state_sales_trend['Order Date'] = state_sales_trend['Order Date'].dt.to_timestamp()

        # Unified View: Sales Trends by State
        fig = px.line(state_sales_trend, x='Order Date', y='Sales', color='State',
            title="Overall Sales Trend by State",
            labels={'Order Date': 'Date', 'Sales': 'Sales (USD)', 'State': 'State'},
                template="plotly_white")

        st.plotly_chart(fig)

        st.markdown("""
        - **Purpose**: 
            - Analyze sales trends across all regions over time.
        - **Visualization**: 
            - Line graph showing monthly sales trends for each region.
        - **Insights**:
            - Compare sales performance across regions.
            - Identify regions with consistent growth or seasonal fluctuations.
        - **Actionable Use**:
            - Develop region-specific strategies based on performance trends.
            - Allocate resources to regions with high growth potential.
        """)
        
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
