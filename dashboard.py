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
# Import machine learning libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split

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
        
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
        df['Month_order'] = df['Order Date'].dt.to_period('M').astype(str)
        df['Year_order'] = df['Order Date'].dt.to_period('Y').astype(str)
        
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

# theme toggle
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
                background-color: #cce2ec; /* Light blue background */
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
page = st.sidebar.radio("Go to", ["🗒️Introduction", "📊 Sales Performance Analysis", "🔮Predictive Model"])

# Introduction page
if page == "🗒️Introduction":
    st.markdown("""
    <h1 style='text-align: center;'> Superstore Sales Perfomance Analysis </h1>"
    <h2 style="text-align: center;">🛍️🛒💳</h2>""" , unsafe_allow_html=True)
    
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

    st.subheader("💥POP QUIZ💥")
    quiz = st.selectbox("Guess the highest revenue category :", ["Furniture", "Technology", "Office Supplies"])
    if quiz == "Technology":
        st.success("Correct! Technology is the highest revenue category.")
        st.image("congrats.jpg", use_container_width=True)

    else:
        st.error("Sorry, answer is incorrect. Please try again.")
        st.image("wrong.jpg", use_container_width=True)

    st.subheader("📋 Dataset Overview")
    st.write(df.head())
    st.write(f"Total records: {len(df)}")
    st.write(f"Total variables: {len(df.columns)}")
    st.write(f"Time range: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")

    with st.expander("ℹ️ Dataset Information"):
        st.write(f"**Dataset Features and Their Description:**")
        # Define descriptions for each column
        column_descriptions = {
            'Order ID': 'The unique identifier for the order',
            'Order Date': 'The date when the order was placed',
            'Ship Date': 'The date when the order was shipped',
            'Ship Mode': 'The mode of shipment for the order',
            'Customer ID': 'The unique identifier for the customer',
            'Customer Name': 'The name of the customer',
            'Segment': 'The customer segment (e.g., Consumer, Corporate, Home Office)',
            'Country': 'The country where the order was placed',
            'City': 'The city where the order was placed',
            'State': 'The state where the order was placed',
            'Postal Code': 'The postal code of the customer',
            'Region': 'The geographical region of the customer',
            'Product ID': 'The unique identifier for the product',
            'Category': 'The category of the product sold',
            'Sub-Category': 'The sub-category of the product sold',
            'Product Name': 'The name of the product sold',
            'Sales': 'The revenue generated from the sale',
            'Month_order': 'The month when the order was placed',
            'Year_order': 'The year when the order was placed',
        }
        
        # Create a DataFrame with column names and their descriptions
        info_df = pd.DataFrame({
            'Column Name': df.columns,
            'Description': [column_descriptions[col] for col in df.columns],
        })

        st.write(info_df)

    with st.expander("**✍️ Made By:**"):
        st.write("""
        **Name: Sara Fuah Jin-Yin**                                                      
        - **Student ID:** 0136704                                         
        - **Email:** 0136704@student.uow.edu.my
                 
        **Name: Teh Yu Kang**
        - **Student ID:** 0136488
        - **Email:** 0136488@student.uow.edu.my
                 
        **Name: Tan Jo Shen**
        - **Student ID:** 0136733
        - **Email:** 0136733@student.uow.edu.my
         
        """)
                 

# Visualizations page
elif page == "📊 Sales Performance Analysis":
    st.title("📊 Sales Performance Analysis")
    
    #Selection box for analysis type
    analysis_type = st.selectbox("View Sales Based On :", ["⌛Timeline", "📚Category", "🌍Geographical Location"])
    
    # Visualization 1 - Line graph for sales over time
    if analysis_type == "⌛Timeline":
    
        st.subheader("Line Graph for Sales Over Time")
        # Time granularity selection
        time_granularity = st.selectbox("Select Time Granularity:", ["Monthly", "Yearly"])
        with st.expander("**ℹ️Why Line Graph?**"):
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
    
        # Group data by Monthly or Yearly
        if time_granularity == "Monthly":
            sales_over_time = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
            sales_over_time['Order Date'] = sales_over_time['Order Date'].dt.to_timestamp()
            # Plot line graph
            fig = px.line(sales_over_time, x='Order Date', y='Sales', 
                          title=f"{time_granularity} Sales Trend", 
                          labels={'Order Date': 'Date', 'Sales': 'Sales (USD)'})
            st.plotly_chart(fig)

            st.markdown("""
            ### Monthly Sales Overview
            - **Seasonality**: 
                - Sales peak during the holiday season (November and December).
                - Sales trough in the beginning of the year (January and February).
                - Indicates a seasonal sales pattern.
            - **Volatility**: 
                - Frequent and large fluctuation .
                - Indicates a highly volatile sales pattern.
            - **Trend**:   
                - Overall upward trend in sales over the months.
                - Indicates a positive growth trajectory.
            """)
        else:  
            sales_over_time = df.groupby(df['Order Date'].dt.to_period('Y'))['Sales'].sum().reset_index()
            sales_over_time['Order Date'] = sales_over_time['Order Date'].dt.to_timestamp()
            # Plot line graph
            fig = px.line(sales_over_time, x='Order Date', y='Sales', 
                        title=f"{time_granularity} Sales Trend", 
                        labels={'Order Date': 'Date', 'Sales': 'Sales (USD)'})
            st.plotly_chart(fig)

            st.markdown(""" 
            ### Yearly Sales Overview
            - **Volatility**: 
                - Low volatility.
                - Indicates smooth and consistent sales pattern.
            - **Trend**:   
                - Overall upward trend in sales over the years after the 2016 dip.
                - Indicates a positive growth trajectory.
            """)


    
    #category visualization 
    elif analysis_type == "📚Category":
        st.subheader("Sales Distribution per Category")

        with st.expander("**ℹ️Why Pie Chart?**"):
            st.markdown(f"""
            ### Sales Distribution Pie Chart
            - **Purpose**: 
                - Analyze sales distribution among categories over time to identify patterns.
            - **Visualization**:
                - Display sales over the categories.
                - Colour coded segments for each category.
            - **Insights**:
                - Highlights gaps between each category.
                - Helps identify the highest and lowest sales categories.
            - **Actionable Use**:
                - Inform strategic decisions for inventory management, marketing campaigns, and resource allocation.
            """)
            
        # Group by category and sum sales, then sort
        Top_category = df.groupby("Category")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)

        # Find total sales across all categories
        total_revenue_category = Top_category["Sales"].sum()

        # Convert the total revenue to an integer, then string, then add '$' sign
        total_revenue_category = f"${int(total_revenue_category)}"

        # Pie chart for 3 categories
        plt.rcParams["figure.figsize"] = (13, 5)  # Width and height of figure in inches
        plt.rcParams['font.size'] = 12.0
        plt.rcParams['font.weight'] = 6

        def autopct_format(values):
            def my_format(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f"${val:,}"  # Format as currency
            return my_format

        colors = ['#BC243C', '#FE840E', '#C62168']
        explode = (0.05, 0.05, 0.05)
        fig1, ax1 = plt.subplots()
        ax1.pie(Top_category['Sales'], colors=colors, labels=Top_category['Category'],
                autopct=autopct_format(Top_category['Sales']), startangle=90, explode=explode)
        centre_circle = plt.Circle((0, 0), 0.82, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        ax1.axis('equal')
        label = ax1.annotate('Total Sales \n' + str(total_revenue_category), color='red', xy=(0, 0), fontsize=12, ha="center")
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)

        st.markdown(f"""
        **Sales Distribution Overview:**  
        - This pie chart illustrates the distribution of sales across product categories.  
        - Technology sales lead the chart, showing the highest demand in this sector.  
        - Office Supplies and Furniture closely follow, indicating a balanced distribution of sales across categories.  

        - 🥇 **Highest Distribution of Sales:** Technology with **${827_456:,}** in total sales.  
        - 🥈 **Second Highest Distribution of Sales:** Furniture with **${728_659:,}** in total sales.  
        - 🥉 **Lowest Distribution of Sales:** Office Supplies with **${705_422:,}** in total sales.  
        """, unsafe_allow_html=True)

        with st.expander("**Detailed Sales Distribution per Category**"):
            # Sort both category and sub-category as per sales
            Top_subcat = df.groupby(["Category", "Sub-Category"])["Sales"].sum().reset_index()
            Top_subcat = Top_subcat.sort_values("Sales", ascending=False).head(10)  # Sort and get top 10
            Top_subcat["Sales"] = Top_subcat["Sales"].astype(int)  # Cast Sales column to integer
            Top_subcat = Top_subcat.sort_values("Category").reset_index(drop=True)

            # Calculate the total sales of all categories
            Top_subcat_1 = Top_subcat.groupby("Category")["Sales"].sum().reset_index()

            outer_colors = ['#FE840E', '#009B77', '#BC243C']  # Outer colors of the pie chart
            inner_colors = ['Orangered', 'tomato', 'coral', "darkturquoise", "mediumturquoise",
                            "paleturquoise", "lightpink", "pink", "hotpink", "deeppink"]  # Inner colors

            # Create the figure and axis
            plt.rcParams["figure.figsize"] = (15, 10)
            fig, ax = plt.subplots()
            ax.axis('equal')
            width = 0.1
            pie = ax.pie(Top_subcat_1['Sales'], radius=1, labels=Top_subcat_1['Category'],
                         colors=outer_colors, wedgeprops=dict(edgecolor='w'))

            # The inner pie chart (Sub-Category level)
            pie2 = ax.pie(Top_subcat['Sales'], radius=1 - width, labels=Top_subcat['Sub-Category'],
                          autopct=autopct_format(Top_subcat['Sales']), labeldistance=0.7,
                          colors=inner_colors, wedgeprops=dict(edgecolor='w'), pctdistance=0.53, rotatelabels=True)

            fraction_text_list = pie2[2]
            for text in fraction_text_list:
                text.set_rotation(315)

            centre_circle = plt.Circle((0, 0), 0.6, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)

            # Ensure equal aspect ratio
            ax.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(f"""
                **Sales Distribution Overview:** 
                - This pie chart illustrates the detailed distribution of sales across product categories and sub-categories. 
                - Phone sales lead the chart, showing the highest demand across all sub-categories. 
                - Chair sales closely follow.
                - Big gap between the top 2 sub-categories and the rest, indicating a significant difference in demand.
                        
                - 🥇 **Highest Distribution of Sales:** Phones with **${327_782:,}** in total sales.
                - 🥈 **Second Highest Distribution of Sales:** Chairs with **${322_822:,}** in total sales.
                - 🥉 **Third Highest Distribution of Sales:** Tables with **${202_810:,}** in total sales.
             """, unsafe_allow_html=True)

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
                    plt.legend()

                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
                st.write(f"""
                         **{category} Sales Overview:** 
                         - This graph illustrates the sales trend for `{category}` over time. 
                         - Highest sales recorded: `{peak_sales:,.0f}` on `{peak_date.strftime('%Y-%m-%d')}`.
                         - Observe seasonal fluctuations and peaks to understand demand variations.""")

    # Visualization 3 - Sales by Geographical Location
    elif analysis_type == "🌍Geographical Location":    
        st.subheader("Sales by States")
        state = ['Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 
                'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 
                'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 
                'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
                'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
                'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

        state_code = ['AL','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
                    'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD',
                    'TN','TX','UT','VT','VA','WA','WV','WI','WY']

        state_df = pd.DataFrame({'State Code': state_code, 'State': state})

        numeric_columns = df.select_dtypes(include=['number']).columns
        sales = df.groupby("State")[numeric_columns].sum().sort_values("Sales", ascending=False)

        sales.reset_index(inplace=True)

        if 'Postal Code' in sales.columns:
            sales.drop('Postal Code', axis=1, inplace=True)

        sales = sales.sort_values('State', ascending=True).reset_index(drop=True)
        sales = sales.merge(state_df, on="State", how="left")
        sales['text'] = sales['State'] + '<br>Sales: ' + sales['Sales'].astype(str)

        # Create the Choropleth map
        fig = go.Figure(data=go.Choropleth(
            locations=sales['State Code'],  # Spatial coordinates
            text=sales['text'],
            z=sales['Sales'].astype(float),  
            locationmode='USA-states',  
            colorscale='Blues',
            colorbar_title="Sales",
        ))

        fig.update_layout(
            geo_scope='usa',  
        )
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

        st.subheader("Sales Trend Analysis by State")
        
        states = df['State'].unique()
        selected_state = st.selectbox("Select a state to view sales trend:", states)
        state_data = df[df['State'] == selected_state]

        state_sales_trend = state_data.groupby(state_data['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        state_sales_trend['Order Date'] = state_sales_trend['Order Date'].dt.to_timestamp()
        peak_sales = state_sales_trend['Sales'].max()
        peak_date = state_sales_trend.loc[state_sales_trend['Sales'].idxmax(), 'Order Date'].strftime('%Y-%m-%d')
        trough_sales = state_sales_trend['Sales'].min()
        trough_date = state_sales_trend.loc[state_sales_trend['Sales'].idxmin(), 'Order Date'].strftime('%Y-%m-%d')
        # Plot line graph for sales trend by state
        fig = px.line(state_sales_trend, x='Order Date', y='Sales',
                title=f"Sales Trend for {selected_state} State",
                labels={'Order Date': 'Date', 'Sales': 'Sales (USD)'},
                template="plotly_white")
        st.plotly_chart(fig)

        # Display dynamic description
        st.markdown(f"""
        ### {selected_state} Sales Overview:
        - This graph illustrates the sales trend for `{selected_state}` over time.
        - Highest sales recorded: **${peak_sales:,.0f}** on `{peak_date}`.
        - Lowest sales recorded: **${trough_sales:,.0f}** on `{trough_date}`.
        - Observe seasonal fluctuations and peaks to understand demand variations.
        """)

        st.markdown(""" 
        - **Purpose**: 
            - Analyze sales trends for the selected state over time.
        - **Visualization**: 
            - Line graph showing monthly sales trends for the selected state.
        - **Insights**:
            - Identify periods of high or low sales performance in the state.
            - Understand seasonal trends and growth opportunities specific to the state.
        - **Actionable Use**:
            - Tailor marketing and sales strategies to the state's performance trends.
            - Optimize inventory and resource allocation for the state.
        """)

        # Overall sales trend by region
        st.subheader("Overall Sales Trend by Region")
        
        # Set default selection to None
        selected_states = st.multiselect("Select States to View Trends:", options=df['State'].unique(), default=[])

        if not selected_states:
            st.warning("Please select at least one state to view trends.")
        else:
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
    st.title("🔮 Sales Prediction Tool")

    # Load data with error handling
    @st.cache_data
    def load_prediction_data():
        try:
            df = pd.read_csv('train.csv')
        
            # Convert date columns to datetime format
            df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
            df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
            df['Month_order'] = df['Order Date'].dt.to_period('M').astype(str)
            df['Year_order'] = df['Order Date'].dt.to_period('Y').astype(str)
            df['Postal Code'] = df['Postal Code'].fillna(5401)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    df = load_prediction_data()

    if df.empty:
        st.warning("No data available. Please check:")
        st.markdown("""
        - train.csv exists in the current directory
        - File contains 'Order Date' and 'Ship Date' columns
        - You're using the correct Kaggle dataset
        """)
        st.stop()

    # Step 1: Preprocess data for supervised models
    @st.cache_data
    def preprocess_data(df):
        df = df.copy()
        # Drop unnecessary columns, but keep 'Sales' and other essential columns
        columns_to_drop = ['Row ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name', 'Order ID']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:  # Only drop columns if they exist
            df.drop(columns=existing_columns, inplace=True)
    
        # Encode categorical variables using LabelEncoder
        categorical_columns = ['Sub-Category', 'Segment', 'Ship Mode', 'Category', 'Region', 'State', 'Country']
        existing_categorical_columns = [col for col in categorical_columns if col in df.columns]
        if existing_categorical_columns:
            for col in existing_categorical_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        # Drop datetime columns
        datetime_columns = df.select_dtypes(include=['datetime', 'datetimetz']).columns
        df.drop(columns=datetime_columns, inplace=True, errors='ignore')

        # Ensure all columns are numeric where applicable
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        # Fill missing values in numeric columns with 0
        df[numeric_columns] = df[numeric_columns].fillna(0)
    
        return df

    processed_df = preprocess_data(df)
    
    # Ensure the processed dataset is not empty
    if processed_df.empty:
        st.error("Processed dataset is empty. Please check your data preprocessing steps.")
        st.stop()
    
    # Split features and target
    X = processed_df.drop(columns=['Sales'])
    
    # Ensure all columns in X are numeric
    X = pd.get_dummies(X, drop_first=True)
    
    y = np.log1p(processed_df['Sales'])  # Log-transform for skewed data

    # Step 2: Train models and cache them
    @st.cache_resource
    def train_models(X_train, y_train):
        # Ensure data is numeric
        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
    
        models = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }
        for model in models.values():
            model.fit(X_train, y_train)
        return models

    # Split data and train models
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(float),  # Ensure X is numeric
        y.astype(float),  # Ensure y is numeric
        test_size=0.2,
        random_state=42
    )
    trained_models = train_models(X_train, y_train)

    # Step 3: Process user input
    def process_user_input(input_data, X_columns):
        input_processed = preprocess_data(input_data)
        input_processed = input_processed.reindex(columns=X_columns, fill_value=0)
        return input_processed

    # Step 4: Evaluate models
    def evaluate_models(models, X_test, y_test):
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = {"RMSE": rmse, "MAE": mae}
        return results

    # Step 5: Create input form
    st.subheader("Enter Product Details for Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
    
        with col1:
            postal_code = st.selectbox("Postal Code", df['Postal Code'].unique())
            sub_category = st.selectbox("Sub-Category", df['Sub-Category'].unique())
    
        with col2:
            segment = st.selectbox("Customer Segment", df['Segment'].unique())
            ship_mode = st.selectbox("Shipping Mode", df['Ship Mode'].unique())
    
        model_choice = st.selectbox("Select Model", list(trained_models.keys()))
        submit_button = st.form_submit_button("Predict Sales")

    # Step 6: Handle prediction
    if submit_button:
        # Validate inputs
        if postal_code not in df['Postal Code'].unique():
            st.warning("Invalid Postal Code. Please enter a valid Postal Code.")
            st.stop()
    
        if sub_category not in df['Sub-Category'].unique():
            st.warning("Invalid Sub-Category. Please select a valid option.")
            st.stop()
    
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Postal Code': [postal_code],
            'Sub-Category': [sub_category],
            'Segment': [segment],
            'Ship Mode': [ship_mode]
        })
    
        # Preprocess user input
        input_processed = process_user_input(input_data, X.columns)
    
        # Predict
        model = trained_models[model_choice]
        prediction = model.predict(input_processed)
        predicted_sales = np.expm1(prediction[0])  # Reverse log transformation
    
        # Display result
        st.success(f"Predicted Sales: **${predicted_sales:,.2f}** using {model_choice}")
# Footer
st.sidebar.markdown("---")
