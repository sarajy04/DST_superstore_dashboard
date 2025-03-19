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

# Add a theme toggle
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            /* Dark theme styling */
            .stApp {
                background-color: #1b2e5e; /* Dark blue background */
                color: #ecf0f1; /* Light gray text */
            }
            /* Sidebar styling */
            .css-1d391kg {
                background-color: #2c3e50; /* Dark blue sidebar */
                color: #ecf0f1; /* Light gray text */
            }
            /* Metric styling */
            div[data-testid="metric-container"] {
                background-color: #34495e; /* Slightly lighter dark blue */
                border: 1px solid #bdc3c7; /* Border around metrics */
                border-radius: 10px; /* Rounded corners */
                padding: 10px; /* Add padding */
                            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            /* Light theme styling */
            .stApp {
                background-color: #f4f4f9; /* Light gray background */
                color: #2c3e50; /* Dark blue text */
            }
            /* Sidebar styling */
            .css-1d391kg {
                background-color: #ffffff; /* White sidebar */
                color: #2c3e50; /* Dark blue text */
            }
            /* Metric styling */
            div[data-testid="metric-container"] {
                background-color: #ecf0f1; /* Light gray background for metrics */
                border: 1px solid #bdc3c7; /* Border around metrics */
                border-radius: 10px; /* Rounded corners */
                padding: 10px; /* Add padding */
                           }
        </style>
    """, unsafe_allow_html=True)

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
    analysis_type = st.selectbox("View Sales Based On :", ["⌛Timeline", "📚Category", "🌍Geographical Location", "💯 All of the Above"])
    
    # Visualization 1 - Line graph for sales over time
    if analysis_type == "⌛Timeline":
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
    elif analysis_type == "📚Category":
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

        # Helper function to format autopct values
        def autopct_format(values):
            def my_format(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f"${val:,}"  # Format as currency
            return my_format
        
        st.markdown(f"""
        **Sales Distribution Overview:**  
        - This pie chart illustrates the distribution of sales across product categories.  
        - Technology sales lead the chart, showing a highest demand in this sector.  
        - Office Supplies and Furniture closely follow, indicating a balanced distribution of sales across categories.  

        - 🥇 **Highest Distribution of Sales:** Technology with **${827_456:,}** in total sales.  
        - 🥈 **Second Highest Distribution of Sales:** Furniture with **${728_659:,}** in total sales.  
        - 🥉 **Lowest Distribution of Sales:** Office Supplies with **${705_422:,}** in total sales.  
        """, unsafe_allow_html=True)
        

        with st.expander("**Detailed Sales Distribution per Category**"):
            # Sort both category and sub-category as per sales
            Top_subcat = df.groupby(["Category", "Sub-Category"])["Sales"].sum().reset_index()
            Top_subcat = Top_subcat.sort_values("Sales", ascending=False).head(10)  # Sort and get top 10
            # Cast Sales column to integer data type
            Top_subcat["Sales"] = Top_subcat["Sales"].astype(int)
            # Sort values by Category
            Top_subcat = Top_subcat.sort_values("Category")
            # Reset index
            Top_subcat.reset_index(drop=True, inplace=True)
    
            # Calculate the total sales of all categories
            Top_subcat_1 = Top_subcat.groupby("Category")["Sales"].sum().reset_index()

            # Define colors
            outer_colors = ['#FE840E', '#009B77', '#BC243C']  # Outer colors of the pie chart
            inner_colors = ['Orangered', 'tomato', 'coral', "darkturquoise", "mediumturquoise",
                    "paleturquoise", "lightpink", "pink", "hotpink", "deeppink"]  # Inner colors

            # Create the figure and axis
            plt.rcParams["figure.figsize"] = (15, 10)  # Set figure size
            fig, ax = plt.subplots()
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

            # Draw the outer pie chart (Category level)
            width = 0.1
            pie = ax.pie(Top_subcat_1['Sales'], radius=1, labels=Top_subcat_1['Category'],
                 colors=outer_colors, wedgeprops=dict(edgecolor='w'))

            # Draw the inner pie chart (Sub-Category level)
            pie2 = ax.pie(Top_subcat['Sales'], radius=1 - width, labels=Top_subcat['Sub-Category'],
                  autopct=autopct_format(Top_subcat['Sales']), labeldistance=0.7,
                  colors=inner_colors, wedgeprops=dict(edgecolor='w'), pctdistance=0.53, rotatelabels=True)

            # Rotate fractions (autopct values)
            fraction_text_list = pie2[2]  # Get the percentage text objects
            for text in fraction_text_list:
                text.set_rotation(315)  # Rotate the autopct values

            # Add a white circle in the center to create a donut chart effect
            centre_circle = plt.Circle((0, 0), 0.6, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)

            # Ensure equal aspect ratio
            ax.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"""
                **Sales Distribution Overview:** 
                - This pie chart illustrates the detail distribution of sales across product categoriesand sub-categories. 
                - Phone sales lead the chart, showing the highest demand across all sub-categories. 
                - Chair sales closely follows.
                - Big gap between the top 2 sub-categories and the rest, indicating a significant difference in demand.
                        
                - 🥇 **Highest Distribution of Sales:** Phones with **${327_782:,}** in total sales.
                - 🥈 **Second Highest Distribution of Sales:** Chairs with **${322_822:,}** in total sales.
                - 🥉 **Third Highest Distribution of Sales:** Tables with **${202_810:,}** in total sales.
             """, unsafe_allow_html=True )

        st.subheader("Sales Trends per Product Category")

        # Category selection widgets
        category_level = st.selectbox("Select Analysis Level:", ["Category", "Sub-Category"])

        if category_level == "Category":
            selected_category = st.selectbox("Select Category:", df['Category'].unique())
            filtered_data = df[df['Category'] == selected_category]
            group_col = 'Category'

        else:
            selected_category = st.selectbox("Select Sub-Category:", df['Sub-Category'].unique())
            filtered_data = df[df['Sub-Category'] == selected_category]
            group_col = 'Sub-Category'

        # Create Line Graph for Selected Category or Subcategory
        if st.button("Generate Sales Trend"):
            if filtered_data.empty:
                st.warning("No data available for selected filters")
            else:
                # Group data by selected level and Order Date
                grouped_data = filtered_data.groupby([group_col, 'Order Date']).agg({'Sales': 'sum'}).reset_index()

                # Generate Line Graphs for the selected category or subcategory
                categories = grouped_data[group_col].unique()

                plt.figure(figsize=(15, len(categories) * 5))  # Adjust figure size dynamically

                for i, category in enumerate(categories, 1):
                 # Filter data for the current category or subcategory
                 category_data = grouped_data[grouped_data[group_col] == category]

                 # Plot line graph for Sales over time
                plt.subplot(len(categories), 1, i)
                plt.plot(category_data['Order Date'], category_data['Sales'], label=category, color='blue', alpha=0.7)

                 # Highlight peaks (top sales value)
                peak_idx = category_data['Sales'].idxmax()
                peak_date = category_data.loc[peak_idx, 'Order Date']
                peak_sales = category_data.loc[peak_idx, 'Sales']
                plt.scatter(peak_date, peak_sales, color='red', s=100, zorder=5)
                plt.text(peak_date, peak_sales + 10, f"Peak: {peak_sales:,.0f}", fontsize=8, ha='center')

                plt.title(f"{category} Sales Trends", fontsize=14)
                plt.xlabel("Order Date", fontsize=12)
                plt.ylabel("Sales", fontsize=12)
                plt.grid(alpha=0.3)

             # Optional: Add legend
                plt.legend()

            plt.tight_layout()  # Ensures that plots don't overlap
            st.pyplot(plt)
            plt.close()
            st.write(f"""
                     **{category} Sales Overview:** 
                     - This graph illustrates the sales trend for `{category}` over time. 
                     - Highest sales recorded:`{peak_sales:,.0f} on `{peak_date.strftime('%Y-%m-%d')}.
                     - Observe seasonal fluctuations and peaks to understand demand variations.""")

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
