import pandas as pd

# Load the dataset
def load_data(file_path):
    """Load data from a CSV file"""
    return pd.read_csv(file_path)

safaricom_data = load_data('data\processed\Safaricom_churn_data.csv')

# Create engagement score
def create_engagement_score(df):
    """Create a feature for overall engagement"""
    df['Engagement Score'] = (df['Monthly Data Usage (MB)'] + 
                              df['Call Duration (Minutes)'] + 
                              df['SMS Sent'] + 
                              df['M-Pesa Transactions'])
    return df

# Average monthly usage (data, call, SMS)
def create_avg_monthly_usage(df):
    """Calculate the average monthly data, call, and SMS usage"""
    df['Avg Monthly Data Usage (MB)'] = (
        df['Last Month Data Usage (MB)'] +
        df['Second Last Month Data Usage (MB)'] +
        df['Third Last Month Data Usage (MB)']
    ) / 3

    df['Avg Call Duration (Minutes)'] = (
        df['Last Month Call Duration (Minutes)'] +
        df['Second Last Month Call Duration (Minutes)'] +
        df['Third Last Month Call Duration (Minutes)']
    ) / 3

    df['Avg SMS Sent'] = (
        df['Last Month SMS Sent'] +
        df['Second Last Month SMS Sent'] +
        df['Third Last Month SMS Sent']
    ) / 3

    return df

# Calculate monthly spend
def create_monthly_spend(df):
    """Create a feature for customer monthly spend"""
    df['Monthly Spend'] = (
        df['Monthly Data Usage (MB)'] * df['Data Rate per MB'] +
        df['Call Duration (Minutes)'] * df['Call Rate per Minute'] +
        df['SMS Sent'] * df['SMS Rate per Message']
    )
    return df

# Previous churn history
def create_previous_churn(df):
    """Indicates whether the customer has churned before"""
    df['Previous Churn'] = df['Churn History'].apply(lambda x: 1 if x > 0 else 0)
    return df

# Churn probability
def create_churn_probability(df):
    """Estimate churn probability based on churn indicators"""
    df['Churn Probability'] = (df['Churn Indicator'] + df['Churn History']) / 2
    return df

# Tenure in months
def create_tenure_in_months(df):
    """Convert tenure from years to months"""
    df['Tenure (Months)'] = df['Tenure (Years)'] * 12
    return df

# Customer interaction ratio
def create_customer_interaction_ratio(df):
    """Calculate interaction ratio based on customer service interactions and tenure"""
    df['Customer Interaction Ratio'] = df['Customer Service Interactions'] / df['Tenure (Months)']
    return df

# Encoding for categorical variables (Region and Gender)
def encode_region_gender(df):
    """Encode categorical variables like Region and Gender"""
    df['Region Encoding'] = df['Region'].astype('category').cat.codes
    df['Gender Encoding'] = df['Gender'].map({'Male': 1, 'Female': 0})
    return df

# Promotions received
def create_promotions_received(df):
    """Flag customers who received promotions"""
    df['Promotions Received'] = df['Promotions'].apply(lambda x: 1 if x > 0 else 0)
    return df

# Region-based features
def create_region_based_features(df, high_churn_regions):
    """Flag customers in high churn regions"""
    df['High Churn Region'] = df['Region'].apply(lambda x: 1 if x in high_churn_regions else 0)
    return df

# Apply all feature engineering functions
def apply_feature_engineering(df):
    """Apply all feature engineering steps"""
    df = create_engagement_score(df)
    df = create_avg_monthly_usage(df)
    df = create_monthly_spend(df)
    df = create_previous_churn(df)
    df = create_churn_probability(df)
    df = create_tenure_in_months(df)
    df = create_customer_interaction_ratio(df)
    df = encode_region_gender(df)
    df = create_promotions_received(df)
    df = create_region_based_features(df, high_churn_regions=['RegionA', 'RegionB'])
    
    return df

# Apply the feature engineering
safaricom_engineered = apply_feature_engineering(safaricom_data)

# Save the engineered dataset
safaricom_engineered.to_csv('safaricom_data_engineered.csv', index=False)

# Check the resulting data
print(safaricom_engineered.head())
