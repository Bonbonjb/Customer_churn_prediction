import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Function to load the dataset
def load_data(file_path):
    """
    Loads the dataset from a given file path.
    """
    return pd.read_csv(file_path)

def preprocess_data(df, target, drop_columns=None):
    """
    Preprocesses the dataset: encodes categorical variables, scales features, and splits into train-test sets.
    
    Arguments:
    df: DataFrame - Input dataset
    target: str - The target column name
    drop_columns: list - List of columns to drop (e.g., non-feature columns)
    
    Returns:
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler: 
    Unscaled and scaled data along with the labels and scaler
    """
    if drop_columns:
        df = df.drop(columns=drop_columns)

    # Convert categorical variables to numerical values
    df = pd.get_dummies(df, columns=['Region', 'Subscription Type', 'Payment Method'], drop_first=True)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    X = df.drop(columns=[target])
    y = df[target]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# Function to train the model
def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier on the given training data.
    
    Arguments:
    X_train: array - Scaled training features
    y_train: array - Training labels
    
    Returns:
    model: Trained model
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and prints out accuracy, precision, recall, F1 score, and a classification report.
    
    Arguments:
    model: Trained model
    X_test: array - Scaled test features
    y_test: array - Test labels
    
    Returns:
    None
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Function for hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning using GridSearchCV.
    
    Arguments:
    X_train: array - Scaled training features
    y_train: array - Training labels
    
    Returns:
    best_model: The model with the best parameters
    """
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model

# Function to save the model and scaler
def save_model(model, scaler, model_path, scaler_path):
    """
    Saves the trained model and scaler.
    
    Arguments:
    model: Trained model
    scaler: Scaler object
    model_path: str - File path for saving the model
    scaler_path: str - File path for saving the scaler
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    """
    Plots a confusion matrix using Seaborn.
    
    Arguments:
    y_test: array - True labels
    y_pred: array - Predicted labels
    
    Returns:
    None
    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(model, X_test, y_test):
    """
    Plots the ROC curve for the given model and test data.
    
    Arguments:
    model: Trained model
    X_test: array - Scaled test features
    y_test: array - Test labels
    
    Returns:
    None
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Function to plot feature importance
def plot_feature_importance(model, X):
    """
    Plots the feature importance from the RandomForest model.
    
    Arguments:
    model: Trained model
    X: DataFrame - The feature data before scaling (so it has column names)
    
    Returns:
    None
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Use the original feature names from X_train
    feature_names = X_train.columns

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='green')
    plt.title('Feature Importance for Churn Prediction Model')
    plt.show()
    
    importance = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 6))
    importance.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importance")
    plt.show()

    # Function to plot churn distribution
def plot_churn_distribution(df, y_test):
    plt.figure(figsize=(10, 6))
    df['Churn Indicator'].value_counts().plot(kind='bar', color=['darkgreen', 'lightgreen'])
    sns.countplot(x=y_test)
    plt.title('Churn vs. Non-Churn Customers in Test Data')
    plt.xlabel('Churn Indicator')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Churn', 'Churn'])
    plt.show()

# Function to plot churn probability distribution
def plot_churn_probability_distribution(model, X_test_scaled):
    plt.figure(figsize=(10, 6))
    probs = model.predict_proba(X_test_scaled)[:, 1]
    sns.histplot(probs, bins=10, kde=True, color='green')
    plt.title("Churn Probability Distribution")
    plt.xlabel("Churn Probability")
    plt.ylabel("Frequency")                                                                                         
    plt.show()


# Main workflow
if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data('data\processed\safaricom_data_engineered.csv')
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(data, 'Churn Indicator', ['Customer ID', 'Churn History'])

    # Train the model
    model = train_model(X_train_scaled, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

    # Tune hyperparameters
    best_model = tune_hyperparameters(X_train_scaled, y_train)

    # Save the model and scaler
    save_model(model, scaler, 'churn_model.pkl', 'scaler.pkl')

    # Plot confusion matrix
    y_pred = model.predict(X_test_scaled)
    plot_confusion_matrix(y_test, y_pred)

    # Plot ROC curve
    plot_roc_curve(model, X_test_scaled, y_test)

    # Plot feature importance using the unscaled X_train (DataFrame with column names)
    plot_feature_importance(model, X_train)

    # Plot churn distribution
    plot_churn_distribution(data, y_test)

    # Plot churn probability distribution
    plot_churn_probability_distribution(model, X_test_scaled)