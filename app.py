import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

def preprocess_data(df, scaler):
    # Remove 'Churn Indicator' if present (it's the target variable)
    df = df.drop(columns=['Churn Indicator'], errors='ignore')

    df = pd.get_dummies(df, columns=['Region', 'Subscription Type', 'Payment Method'], drop_first=True)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Remove the non-feature columns
    df = df.drop(columns=['Customer ID'], errors='ignore')
    
    # Ensure all expected columns are present
    expected_columns = scaler.feature_names_in_
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match the order used during training
    df = df[expected_columns]
    
    # Scale the features
    scaled_features = scaler.transform(df)
    return scaled_features

# Function to plot churn distribution
def plot_churn_distribution(df, y_test):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y_test, data=df, palette='coolwarm')
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    st.pyplot(plt)

# Function to plot churn probability distribution
def plot_churn_probability_distribution(model, X_scaled):
    churn_probabilities = model.predict_proba(X_scaled)[:, 1]
    
    plt.figure(figsize=(8, 5))
    sns.histplot(churn_probabilities, bins=20, kde=True, color='green')
    plt.title("Churn Probability Distribution")
    plt.xlabel("Churn Probability")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Function to plot ROC curve
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(plt)

# Function to plot feature importance
def plot_feature_importance(model, X):
    importances = model.feature_importances_
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="Greens_d")
    plt.title('Feature Importance for Churn Prediction Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    st.write("Upload your dataset to predict customer churn.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the data
        data = pd.read_csv(uploaded_file)

        # Feature engineering on uploaded data
        # Display uploaded data
        st.write("Uploaded Data:")
        st.dataframe(data.head())

        # Preprocess data for prediction
        features = preprocess_data(data, scaler)

        # Make predictions
        predictions = model.predict(features)
        prediction_probs = model.predict_proba(features)[:, 1]

        # Display predictions
        data['Predicted Churn'] = predictions
        data['Churn Probability'] = prediction_probs
        st.write("Predictions:")
        st.dataframe(data[['Predicted Churn', 'Churn Probability']])

        # Plot churn distribution
        st.write("Churn Distribution:")
        plot_churn_distribution(data, data['Churn Indicator'])

        # Plot churn probability distribution
        st.write("Churn Probability Distribution:")
        plot_churn_probability_distribution(model, features)

        # Evaluate model performance
        if st.checkbox("Show Model Performance Metrics"):
            y_test = data['Churn Indicator']  # Assuming the true churn indicator is present
            y_pred = predictions
            
            # Display classification report
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred)

            # Plot ROC curve
            plot_roc_curve(y_test, prediction_probs)

        # Feature Importance
        if st.checkbox("Show Feature Importance"):
            st.write("Feature Importance for the model:")
            plot_feature_importance(model, pd.DataFrame(features, columns=scaler.feature_names_in_))

        # Sidebar Information
st.sidebar.header("About")
st.sidebar.write("""
This application predicts customer churn based on demographic and usage data. 
Upload a dataset to predict churn and view model performance metrics such as 
accuracy, confusion matrix, and ROC curve.
""")    

# Run the app
if __name__ == "__main__":
    main()
