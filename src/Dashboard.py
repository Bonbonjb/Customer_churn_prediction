import dash 
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize Dash app
app = dash.Dash(__name__)

# Load the dataset
safaricom_churn_data = pd.read_csv('data\processed\safaricom_data_engineered.csv')

# Data preprocessing
X = safaricom_churn_data.drop(columns=['Customer ID', 'Churn Indicator'])
y = safaricom_churn_data['Churn Indicator']

# One-hot encoding categorical variables
X = pd.get_dummies(X, columns=['Region', 'Subscription Type', 'Payment Method'], drop_first=True)
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Function to create churn distribution figure
def create_churn_distribution_fig(data):
    return px.histogram(data, x='Churn Indicator', title="Churn vs. Non-Churn Distribution")

# Function to create probability distribution figure
def create_probability_distribution_fig(model, X):
    probs = model.predict_proba(X)[:, 1]
    return px.histogram(probs, nbins=10, title="Churn Probability Distribution", labels={'value': 'Churn Probability'})

# Create Plotly figures
churn_dist_fig = create_churn_distribution_fig(safaricom_churn_data)
probability_dist_fig = create_probability_distribution_fig(model, X_test)

# Confusion Matrix Figure
conf_matrix_fig = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=['No Churn', 'Churn'],
    y=['No Churn', 'Churn'],
    colorscale='Blues',
    text=conf_matrix,
    hoverinfo='text',
    showscale=False
))
conf_matrix_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")

# ROC Curve Figure
roc_curve_fig = go.Figure()
roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})'))
roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
roc_curve_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

# Feature Importance Figure
feature_importance_fig = px.bar(feature_importance_df, x='Importance', y='Feature', title="Feature Importance")

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Safaricom Customer Churn Prediction Dashboard"),
    
    # Churn Distribution
    html.Div([
        dcc.Graph(figure=churn_dist_fig)
    ], style={'width': '48%', 'display': 'inline-block'}),

    # Probability Distribution
    html.Div([
        dcc.Graph(figure=probability_dist_fig)
    ], style={'width': '48%', 'display': 'inline-block'}),

    # Confusion Matrix
    html.Div([
        dcc.Graph(figure=conf_matrix_fig)
    ], style={'width': '48%', 'display': 'inline-block'}),

    # ROC Curve
    html.Div([
        dcc.Graph(figure=roc_curve_fig)
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    # Feature Importance
    html.Div([
        dcc.Graph(figure=feature_importance_fig)
    ], style={'width': '48%', 'display': 'inline-block'})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
