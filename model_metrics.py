import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix,precision_recall_curve, auc
from sklearn.model_selection import train_test_split

def evaluate_model():
    # Create directories if not exist
    os.makedirs('./model_visualizations', exist_ok=True)

    # Load the dataset
    df = pd.read_csv('./data/final_data_small.csv')

    # Target and features
    label = 'IS_DROPOFF'
    features = [col for col in df.columns if col != label]

    X = df[features]
    y = df[label]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load models and evaluate
    model_dir = './saved_models'
    results = {}

    for fname in os.listdir(model_dir):
        if not fname.endswith('.joblib'):
            continue

        model_path = os.path.join(model_dir, fname)
        model_name = fname.replace('_model.joblib', '')

        try:
            model = load(model_path)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba)
            prc, rec_curve, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(rec_curve, prc)
            cm = confusion_matrix(y_test, y_pred)

            results[model_name] = {
                'accuracy': acc,
                'roc_auc': roc,
                'pr_auc': pr_auc
            }

            # Save confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f'./model_visualizations/confusion_matrix_{model_name}.png')
            plt.close()

            # Save PR curve
            plt.figure(figsize=(6, 5))
            plt.plot(rec_curve, prc, label=f'PR AUC = {pr_auc:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.savefig(f'./model_visualizations/pr_curve_{model_name}.png')
            plt.close()

        except Exception as e:
            print(f"Error loading {fname}: {e}")

    # Save summary metrics
    results_df = pd.read_csv('./model_results/summary_metrics.csv')
    print(results_df)

if __name__ == '__main__':
    evaluate_model()