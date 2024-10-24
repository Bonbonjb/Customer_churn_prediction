import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
safaricom_data = pd.read_csv('data\processed\Safaricom_churn_data.csv')

# 1. Overview of the dataset
def overview_of_dataset(df):
    print("First five rows of the dataset:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())

# 2. Check for missing values
def check_missing_values(df):
    print("Missing values in the dataset:")
    print(df.isnull().sum())

# 3. Churn Distribution
def visualize_churn_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Churn Indicator', data=df, color='green')
    plt.title('Churn Distribution')
    plt.xlabel('Churn Indicator (0: No, 1: Yes)')
    plt.ylabel('Count')
    plt.show()

# 4. Churn Rate by Age
def visualize_churn_rate_by_age(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn Indicator', y='Age', data=df, color='green')
    plt.title('Churn Rate by Age')
    plt.xlabel('Churn Indicator (0: No, 1: Yes)')
    plt.ylabel('Age')
    plt.show()

# 5. Monthly Data Usage by Churn
def visualize_monthly_data_usage_by_churn(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn Indicator', y='Monthly Data Usage (MB)', data=df, color='green')
    plt.title('Monthly Data Usage by Churn Status')
    plt.xlabel('Churn Indicator (0: No, 1: Yes)')
    plt.ylabel('Monthly Data Usage (MB)')
    plt.show()

# 6. Correlation Heatmap
def visualize_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Calling the functions to perform EDA
overview_of_dataset(safaricom_data)             # Dataset overview
check_missing_values(safaricom_data)            # Missing values check
visualize_churn_distribution(safaricom_data)    # Churn distribution visualization
visualize_churn_rate_by_age(safaricom_data)     # Churn rate by age
visualize_monthly_data_usage_by_churn(safaricom_data) # Monthly data usage by churn
visualize_correlation_heatmap(safaricom_data)   # Correlation heatmap