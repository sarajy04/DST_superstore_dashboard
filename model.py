import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: The file 'train.csv' was not found.")
    exit()

# Step 1: Data Preprocessing
df.drop(columns=['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 
                 'Product Name', 'Order Date', 'Ship Date', 'Country', 'City', 'State'], inplace=True)

# Handle missing values
df['Postal Code'] = df['Postal Code'].fillna(df['Postal Code'].median())

# Encode categorical variables
categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop non-numeric columns dynamically
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
df.drop(columns=non_numeric_cols, inplace=True)

# Separate features and target variable
X = df.drop(columns=['Sales'])
y = df['Sales']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Debugging: Check dataset shape
print("Shape of X_scaled:", X_scaled.shape)

# Split into training and testing sets for supervised models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Train Supervised Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression()
}

supervised_results = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    supervised_results[model_name] = {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# Step 3: Train Unsupervised Models
# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=5, random_state=42)
clusters_gmm = gmm.fit_predict(X_scaled)

# Evaluate K-Means and GMM
silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, clusters_kmeans)
calinski_harabasz_kmeans = calinski_harabasz_score(X_scaled, clusters_kmeans)

silhouette_gmm = silhouette_score(X_scaled, clusters_gmm)
davies_bouldin_gmm = davies_bouldin_score(X_scaled, clusters_gmm)
calinski_harabasz_gmm = calinski_harabasz_score(X_scaled, clusters_gmm)

unsupervised_results = {
    "K-Means": {
        "Silhouette Score": silhouette_kmeans,
        "Davies-Bouldin Index": davies_bouldin_kmeans,
        "Calinski-Harabasz Index": calinski_harabasz_kmeans
    },
    "Gaussian Mixture Model": {
        "Silhouette Score": silhouette_gmm,
        "Davies-Bouldin Index": davies_bouldin_gmm,
        "Calinski-Harabasz Index": calinski_harabasz_gmm
    }
}

# Step 4: Compare Results
print("\n--- Supervised Model Results ---")
for model_name, metrics in supervised_results.items():
    print(f"\nModel: {model_name}")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2: {metrics['R2']:.2f}")

print("\n--- Unsupervised Model Results ---")
for model_name, metrics in unsupervised_results.items():
    print(f"\nModel: {model_name}")
    print(f"Silhouette Score: {metrics['Silhouette Score']:.2f}")
    print(f"Davies-Bouldin Index: {metrics['Davies-Bouldin Index']:.2f}")
    print(f"Calinski-Harabasz Index: {metrics['Calinski-Harabasz Index']:.2f}")

# Step 5: Save Results
results_df = pd.DataFrame({
    "Model": list(supervised_results.keys()) + list(unsupervised_results.keys()),
    "MSE": [supervised_results[m]['MSE'] if m in supervised_results else None for m in list(supervised_results.keys()) + list(unsupervised_results.keys())],
    "RMSE": [supervised_results[m]['RMSE'] if m in supervised_results else None for m in list(supervised_results.keys()) + list(unsupervised_results.keys())],
    "R2": [supervised_results[m]['R2'] if m in supervised_results else None for m in list(supervised_results.keys()) + list(unsupervised_results.keys())],
    "Silhouette Score": [unsupervised_results[m]['Silhouette Score'] if m in unsupervised_results else None for m in list(supervised_results.keys()) + list(unsupervised_results.keys())],
    "Davies-Bouldin Index": [unsupervised_results[m]['Davies-Bouldin Index'] if m in unsupervised_results else None for m in list(supervised_results.keys()) + list(unsupervised_results.keys())],
    "Calinski-Harabasz Index": [unsupervised_results[m]['Calinski-Harabasz Index'] if m in unsupervised_results else None for m in list(supervised_results.keys()) + list(unsupervised_results.keys())]
})

results_df.to_csv("model_comparison_results.csv", index=False)
print("\nComparison results saved to 'model_comparison_results.csv'")