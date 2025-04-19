"""
Enhanced Big Data Processor for Hospital Records Mining.
Extends the original BigDataProcessor with additional functionality for detailed results.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get logger
logger = logging.getLogger(__name__)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)

# Import from the local src directory
from src.big_data_processor import BigDataProcessor

class EnhancedBigDataProcessor(BigDataProcessor):
    """Enhanced processor for large-scale hospital records with detailed results."""
    
    def __init__(self, data_dir, output_dir, random_state=42, use_deep_learning=False):
        """Initialize the enhanced big data processor."""
        super().__init__(data_dir, output_dir, random_state, use_deep_learning)
        self.detailed_results = {}
        self.feature_names = None
        self.X_test = None
        self.y_test = None
        self.models = {}
        
    def train_model(self, feature_data):
        """
        Override the train_model method to store detailed results.
        
        Parameters:
        -----------
        feature_data : pandas.DataFrame
            DataFrame with features and drop-off labels
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Select features and target
        target_col = 'IS_DROPOFF'
        if target_col not in feature_data.columns:
            logger.error(f"Target column '{target_col}' not found in feature data")
            return None
        
        drop_cols = ['PATIENT', 'IS_CURRENT_DROPOFF', 'POOR_ADHERENCE', 'HAS_CHRONIC_CONDITION', 'ADHERENCE_RATIO']
        drop_cols = [col for col in drop_cols if col in feature_data.columns]
        drop_cols.append(target_col)
        
        X = feature_data.drop(drop_cols, axis=1)
        y = feature_data[target_col]
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Store test data for feature importance calculation
        self.X_test = X_test
        self.y_test = y_test
        
        logger.info(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")
        logger.info(f"Drop-off rate in training set: {y_train.mean():.2%}")
        logger.info(f"Drop-off rate in testing set: {y_test.mean():.2%}")
        
        # Identify categorical and numeric features
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        numeric_features = [col for col in X.columns if col not in categorical_features]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) if categorical_features else ('cat', 'passthrough', [])
            ]
        )
        
        # Create models
        self.models = {
            'logistic_regression': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ]),
            'random_forest': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=self.random_state, n_estimators=100))
            ]),
            'gradient_boosting': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(random_state=self.random_state, n_estimators=100))
            ])
        }
        
        # Train and evaluate models
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Calculate precision-recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate feature importance if possible
            feature_importance = self.calculate_feature_importance(name, model, X_test, y_test)
            
            # Store detailed metrics
            self.detailed_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'confusion_matrix': cm,
                'feature_importance': feature_importance,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Store metrics for return
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }
            
            # Log results
            logger.info(f"{name} results:")
            logger.info(f"  - Accuracy: {accuracy:.4f}")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            logger.info(f"  - F1 Score: {f1:.4f}")
            logger.info(f"  - ROC AUC: {roc_auc:.4f}")
            logger.info(f"  - PR AUC: {pr_auc:.4f}")
            
            # Print classification report
            logger.info(f"Classification Report for {name}:")
            logger.info(classification_report(y_test, y_pred))
            
            # Log confusion matrix
            logger.info(f"Confusion Matrix for {name}:")
            logger.info(cm)
            
            # Save model visualization
            os.makedirs(os.path.join(self.output_dir, 'model_visualizations'), exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(self.output_dir, 'model_visualizations', f'confusion_matrix_{name}.png'))
            plt.close()
            
            # Plot PR curve
            plt.figure(figsize=(8, 6))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(recall_curve, precision_curve, label=f'PR AUC = {pr_auc:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {name}')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'model_visualizations', f'pr_curve_{name}.png'))
            plt.close()
        
        # Save results to CSV
        results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'model'})
        results_df.to_csv(os.path.join(self.output_dir, 'model_results.csv'), index=False)
        
        # Add deep learning model if enabled
        if self.use_deep_learning:
            logger.info("Deep learning is enabled, training TensorFlow model...")
            dl_results = self.train_deep_learning_model(X_train, X_test, y_train, y_test, preprocessor)
            results['deep_learning'] = dl_results
            
            # For deep learning, we'll create a simpler detailed results structure
            # since the original implementation doesn't return y_pred_proba
            # The deep learning results don't include pr_auc in the returned dictionary
            self.detailed_results['deep_learning'] = {
                'accuracy': dl_results['accuracy'],
                'precision': dl_results['precision'],
                'recall': dl_results['recall'],
                'f1_score': dl_results['f1_score'],
                'roc_auc': dl_results['roc_auc'],
                'pr_auc': 0.0,  # Set a default value since it's not in the original results
                'confusion_matrix': None,  # We don't have this in the original results
                'feature_importance': None,  # Deep learning feature importance is more complex
                'y_pred': None,  # We don't have this in the original results
                'y_pred_proba': None  # We don't have this in the original results
            }
            
            logger.info("Deep learning model training complete")
            
            # Update results CSV with deep learning results
            results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'model'})
            results_df.to_csv(os.path.join(self.output_dir, 'model_results.csv'), index=False)
        else:
            logger.info("Deep learning is disabled, skipping TensorFlow model")
        
        # Identify best model
        best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
        logger.info(f"Best model: {best_model[0]} with ROC AUC = {best_model[1]['roc_auc']:.4f}")
        
        # Return average metrics across all models
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in results.values()]),
            'precision': np.mean([m['precision'] for m in results.values()]),
            'recall': np.mean([m['recall'] for m in results.values()]),
            'f1_score': np.mean([m['f1_score'] for m in results.values()]),
            'roc_auc': np.mean([m['roc_auc'] for m in results.values()]),
            'pr_auc': np.mean([m['pr_auc'] for m in results.values()])
        }
        
        return avg_metrics
    
    def calculate_feature_importance(self, model_name, model, X_test, y_test):
        """
        Calculate feature importance for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : sklearn.pipeline.Pipeline
            Trained model pipeline
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test target
            
        Returns:
        --------
        dict or None
            Dictionary with feature importance information or None if not applicable
        """
        try:
            if model_name == 'random_forest' or model_name == 'gradient_boosting':
                # Get feature names after preprocessing
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']
                
                # Get feature importances
                importances = classifier.feature_importances_
                
                # Get feature names after preprocessing
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    # Fallback for older scikit-learn versions
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                # Create a DataFrame with feature importances
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                
                # Sort by importance
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
                return feature_importance.to_dict('list')
            
            elif model_name == 'logistic_regression':
                # For logistic regression, use permutation importance
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']
                
                # Transform the test data
                X_test_transformed = preprocessor.transform(X_test)
                
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    classifier, X_test_transformed, y_test, 
                    n_repeats=10, random_state=self.random_state
                )
                
                # Get feature names after preprocessing
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    # Fallback for older scikit-learn versions
                    feature_names = [f"feature_{i}" for i in range(len(perm_importance.importances_mean))]
                
                # Create a DataFrame with feature importances
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': perm_importance.importances_mean
                })
                
                # Sort by importance
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
                return feature_importance.to_dict('list')
            
            else:
                return None
        
        except Exception as e:
            logger.warning(f"Could not calculate feature importance for {model_name}: {str(e)}")
            return None
    
    def get_detailed_results(self):
        """
        Get detailed results for all models.
        
        Returns:
        --------
        dict
            Dictionary with detailed results for each model
        """
        return self.detailed_results
