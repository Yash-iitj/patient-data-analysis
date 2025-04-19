import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Setup
os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./model_visualizations', exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
choice_of_data = input("Enter 1 to use smaller dataset, 2 to use bigger dataset: ")
if choice_of_data == 1:
    data_path = './data/final_data_small.csv'
else:
    data_path = './data/final_data_big.csv'
df = pd.read_csv(data_path)

# Add synthetic binary target
df['IS_DROPOFF'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

# Separate features and target
target = 'IS_DROPOFF'
drop_cols = ['IS_DROPOFF']
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y = df[target]

# Identify numeric columns (all in this case)
numeric = X.columns.tolist()

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

# Training and evaluation
for name, model in models.items():
    logger.info(f"Training model: {name}")

    pipeline = Pipeline([
        ('preprocessor', numeric_transformer),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    prc, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec_curve, prc)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"{name} -- Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc:.4f}, PR AUC: {pr_auc:.4f}")
    logger.info(classification_report(y_test, y_pred))

    # Save model
    model_path = f'./saved_models/{name}_model.joblib'
    dump(pipeline, model_path)

    # Save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'./model_visualizations/confusion_matrix_{name}.png')
    plt.close()

    # Save PR Curve
    plt.figure(figsize=(6, 5))
    plt.plot(rec_curve, prc, label=f'PR AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend()
    plt.savefig(f'./model_visualizations/pr_curve_{name}.png')
    plt.close()

    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc,
        'pr_auc': pr_auc
    }

# Save summary
summary_path = './model_visualizations/model_results.csv'
pd.DataFrame(results).T.to_csv(summary_path)
