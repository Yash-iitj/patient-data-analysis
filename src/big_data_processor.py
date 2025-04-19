"""
Big Data Processor for Hospital Records Mining.
Implements non-Spark processing of large-scale hospital data for dropout prediction.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import TensorFlow for deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning models will be skipped.")

logger = logging.getLogger(__name__)

class BigDataProcessor:
    """Processor for large-scale hospital records without using Spark."""
    
    def __init__(self, data_dir, output_dir, random_state=42, use_deep_learning=False):
        """
        Initialize the big data processor.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the big data files
        output_dir : str
            Directory to save the output
        random_state : int
            Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.random_state = random_state
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        
        if use_deep_learning and not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow is not available. Deep learning is disabled.")
        elif use_deep_learning:
            logger.info("Deep learning is enabled for BigDataProcessor")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataframes
        self.patients_df = None
        self.encounters_df = None
        self.conditions_df = None
        self.medications_df = None
        
        # Threshold configurations for dropout definition
        self.gap_thresholds = {
            'ambulatory': 207,
            'emergency': 220,
            'outpatient': 203,
            'wellness': 365,
            'urgentcare': 28,
            'default': 180  # Default for other encounter types
        }
    
    def load_data(self):
        """
        Load the big data files required for processing.
        This method attempts to load the following datasets from the specified 
        data directory (`self.data_dir`):
        - `patients_big.csv`: Contains patient information. If the file exists, 
          it is loaded into `self.patients_df`. Logs the number of records loaded 
          or a warning if the file is not found.
        - `encounters_big.csv`: Contains encounter details. If the file exists, 
          it is loaded into `self.encounters_df`. Logs the number of records loaded 
          or a warning if the file is not found.
        - `conditions_big.csv`: Contains condition data. If the file exists, 
          it is loaded into `self.conditions_df`. Logs the number of records loaded 
          or a warning if the file is not found.
        - `medications_big.csv`: Contains medication data. If the file exists, 
          it is loaded into `self.medications_df`. Logs the number of records loaded 
          or a warning if the file is not found.
        Logs appropriate messages for each dataset, indicating success or failure 
        in loading the respective file.
        """
        """Load the big data files."""
        logger.info("Loading big data files...")
        
        # Load patients data
        patients_path = os.path.join(self.data_dir, 'patients_big.csv')
        if os.path.exists(patients_path):
            self.patients_df = pd.read_csv(patients_path)
            logger.info(f"Loaded {len(self.patients_df)} patients")
        else:
            logger.warning(f"Patients data not found at {patients_path}")
        
        # Load encounters data
        encounters_path = os.path.join(self.data_dir, 'encounters_big.csv')
        if os.path.exists(encounters_path):
            self.encounters_df = pd.read_csv(encounters_path)
            logger.info(f"Loaded {len(self.encounters_df)} encounters")
        else:
            logger.warning(f"Encounters data not found at {encounters_path}")
        
        # Load conditions data
        conditions_path = os.path.join(self.data_dir, 'conditions_big.csv')
        if os.path.exists(conditions_path):
            self.conditions_df = pd.read_csv(conditions_path)
            logger.info(f"Loaded {len(self.conditions_df)} conditions")
        else:
            logger.warning(f"Conditions data not found at {conditions_path}")
        
        # Load medications data if available
        medications_path = os.path.join(self.data_dir, 'medications_big.csv')
        if os.path.exists(medications_path):
            self.medications_df = pd.read_csv(medications_path)
            logger.info(f"Loaded {len(self.medications_df)} medications")
    
    def preprocess_data(self):
        """Preprocess the data for analysis."""
        logger.info("Preprocessing data...")
        
        # Convert date columns to datetime (ensuring timezone consistency)
        if self.encounters_df is not None:
            self.encounters_df['START'] = pd.to_datetime(self.encounters_df['START']).dt.tz_localize(None)
            self.encounters_df['STOP'] = pd.to_datetime(self.encounters_df['STOP']).dt.tz_localize(None)
        
        if self.conditions_df is not None:
            self.conditions_df['START'] = pd.to_datetime(self.conditions_df['START']).dt.tz_localize(None)
            if 'STOP' in self.conditions_df.columns:
                # Handle empty strings in STOP column
                self.conditions_df['STOP'] = pd.to_datetime(
                    self.conditions_df['STOP'].replace('', np.nan), errors='coerce'
                ).dt.tz_localize(None)
        
        if self.medications_df is not None:
            self.medications_df['START'] = pd.to_datetime(self.medications_df['START']).dt.tz_localize(None)
            if 'STOP' in self.medications_df.columns:
                self.medications_df['STOP'] = pd.to_datetime(
                    self.medications_df['STOP'].replace('', np.nan), errors='coerce'
                ).dt.tz_localize(None)
        
        # Calculate patient age if birthdate is available
        if self.patients_df is not None and 'BIRTHDATE' in self.patients_df.columns:
            self.patients_df['BIRTHDATE'] = pd.to_datetime(self.patients_df['BIRTHDATE']).dt.tz_localize(None)
            current_date = pd.Timestamp(datetime.now()).tz_localize(None)
            self.patients_df['AGE'] = (current_date - self.patients_df['BIRTHDATE']).dt.days / 365.25
        
        logger.info("Preprocessing complete")
    
    def identify_dropoffs(self):
        """
        Identify patient drop-offs based on the gap-based definition.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with patient drop-off labels
        """
        logger.info("Identifying patient drop-offs...")
        
        if self.encounters_df is None:
            logger.error("Encounters data not loaded")
            return None
        
        # Sort encounters by patient and start date
        encounters_sorted = self.encounters_df.sort_values(['PATIENT', 'START'])
        
        # Calculate days between visits for each patient
        encounters_sorted['NEXT_VISIT'] = encounters_sorted.groupby('PATIENT')['START'].shift(-1)
        encounters_sorted['DAYS_TO_NEXT_VISIT'] = (encounters_sorted['NEXT_VISIT'] - encounters_sorted['START']).dt.days
        
        # Apply threshold based on encounter class
        encounters_sorted['THRESHOLD'] = encounters_sorted['ENCOUNTERCLASS'].map(
            lambda x: self.gap_thresholds.get(str(x).lower(), self.gap_thresholds['default'])
        )
        
        # Identify potential drop-offs based on gap threshold
        encounters_sorted['IS_POTENTIAL_DROPOFF'] = encounters_sorted['DAYS_TO_NEXT_VISIT'] > encounters_sorted['THRESHOLD']
        
        # Get the last encounter for each patient
        last_encounters = encounters_sorted.sort_values('START').groupby('PATIENT').last().reset_index()
        
        # Calculate days since last encounter to current date
        # Use a timezone-naive datetime to match the START column
        current_date = pd.Timestamp(datetime.now()).tz_localize(None)
        last_encounters['DAYS_SINCE_LAST'] = (current_date - last_encounters['START']).dt.days
        
        # Identify current drop-offs
        last_encounters['IS_CURRENT_DROPOFF'] = last_encounters['DAYS_SINCE_LAST'] > last_encounters['THRESHOLD']
        
        # Identify patients with chronic conditions
        if self.conditions_df is not None:
            # Define chronic condition keywords
            chronic_keywords = ['diabetes', 'hypertension', 'heart', 'asthma', 'copd', 
                               'depression', 'anxiety', 'arthritis', 'chronic']
            
            # Create a condition for each keyword
            has_chronic = []
            for patient in tqdm(last_encounters['PATIENT'], desc="Processing chronic conditions"):
                patient_conditions = self.conditions_df[
                    self.conditions_df['PATIENT'] == patient
                ]['DESCRIPTION'].astype(str).str.lower()
                
                if any(patient_conditions.str.contains(keyword).any() for keyword in chronic_keywords):
                    has_chronic.append(1)
                else:
                    has_chronic.append(0)
            
            last_encounters['HAS_CHRONIC_CONDITION'] = has_chronic
            
            # Adjust threshold for patients with chronic conditions
            last_encounters.loc[last_encounters['HAS_CHRONIC_CONDITION'] == 1, 'THRESHOLD'] = \
                last_encounters.loc[last_encounters['HAS_CHRONIC_CONDITION'] == 1, 'THRESHOLD'] * 0.5
            
            # Recalculate drop-off status with adjusted thresholds
            last_encounters['IS_CURRENT_DROPOFF'] = last_encounters['DAYS_SINCE_LAST'] > last_encounters['THRESHOLD']
        else:
            last_encounters['HAS_CHRONIC_CONDITION'] = 0
        
        # Calculate medication adherence
        if self.medications_df is not None:
            # Calculate medication duration
            self.medications_df['DURATION_DAYS'] = np.where(
                self.medications_df['STOP'].notna(),
                (self.medications_df['STOP'] - self.medications_df['START']).dt.days,
                30  # Default to 30 days for ongoing medications
            )
            
            # Calculate medication adherence
            med_adherence = []
            for patient in tqdm(last_encounters['PATIENT'], desc="Processing medication adherence"):
                patient_meds = self.medications_df[self.medications_df['PATIENT'] == patient]
                if len(patient_meds) > 0:
                    # Calculate expected vs actual refills
                    expected_refills = patient_meds['DURATION_DAYS'].sum() / 30
                    actual_refills = patient_meds['DISPENSES'].sum() if 'DISPENSES' in patient_meds.columns else 0
                    adherence_ratio = actual_refills / expected_refills if expected_refills > 0 else 1
                    med_adherence.append(adherence_ratio)
                else:
                    med_adherence.append(1)  # No medications, assume perfect adherence
            
            last_encounters['ADHERENCE_RATIO'] = med_adherence
            
            # Identify patients with poor medication adherence
            last_encounters['POOR_ADHERENCE'] = last_encounters['ADHERENCE_RATIO'] < 0.8
        else:
            last_encounters['ADHERENCE_RATIO'] = 1
            last_encounters['POOR_ADHERENCE'] = False
        
        # Final drop-off definition: Either gap-based drop-off OR poor medication adherence
        last_encounters['IS_DROPOFF'] = (
            last_encounters['IS_CURRENT_DROPOFF'] | 
            last_encounters['POOR_ADHERENCE']
        )
        
        # Count drop-offs
        dropoff_count = last_encounters['IS_DROPOFF'].sum()
        total_patients = len(last_encounters)
        dropoff_rate = dropoff_count / total_patients if total_patients > 0 else 0
        
        logger.info(f"Identified {dropoff_count} drop-offs out of {total_patients} patients ({dropoff_rate:.2%})")
        
        return last_encounters
    
    def extract_features(self, labeled_data):
        """
        Extract features for the dropout prediction model.
        
        Parameters:
        -----------
        labeled_data : pandas.DataFrame
            DataFrame with patient drop-off labels
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with features for modeling
        """
        logger.info("Extracting features...")
        
        # Start with labeled data
        feature_data = labeled_data.copy()
        
        # Add demographic features from patients data
        if self.patients_df is not None:
            # Select relevant columns
            demographic_cols = ['Id', 'AGE', 'GENDER', 'RACE', 'ETHNICITY']
            demographic_cols = [col for col in demographic_cols if col in self.patients_df.columns]
            
            # Add income and healthcare columns if available
            for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
                if col in self.patients_df.columns:
                    demographic_cols.append(col)
            
            demographic_features = self.patients_df[demographic_cols].copy()
            
            # Rename Id to PATIENT for joining
            demographic_features = demographic_features.rename(columns={'Id': 'PATIENT'})
            
            # Join with feature data
            feature_data = feature_data.merge(demographic_features, on='PATIENT', how='left')
        
        # Add encounter patterns
        if self.encounters_df is not None:
            # Calculate encounter patterns by patient
            encounter_patterns = self.encounters_df.groupby('PATIENT').agg(
                TOTAL_ENCOUNTERS=('Id', 'count'),
                FIRST_ENCOUNTER=('START', 'min'),
                LAST_ENCOUNTER=('START', 'max')
            ).reset_index()
            
            # Calculate encounter duration
            encounter_patterns['PATIENT_HISTORY_DAYS'] = (
                encounter_patterns['LAST_ENCOUNTER'] - encounter_patterns['FIRST_ENCOUNTER']
            ).dt.days
            
            # Calculate encounter frequency
            encounter_patterns['ENCOUNTER_FREQUENCY'] = np.where(
                encounter_patterns['PATIENT_HISTORY_DAYS'] > 0,
                encounter_patterns['TOTAL_ENCOUNTERS'] / encounter_patterns['PATIENT_HISTORY_DAYS'] * 365.25,
                0
            )
            
            # Count encounter types
            encounter_types = self.encounters_df.groupby(['PATIENT', 'ENCOUNTERCLASS']).size().unstack(fill_value=0)
            if not encounter_types.empty:
                encounter_types.columns = [f'ENCOUNTERS_{col.upper()}' for col in encounter_types.columns]
                encounter_types = encounter_types.reset_index()
                
                # Join encounter patterns with encounter types
                encounter_features = encounter_patterns.merge(encounter_types, on='PATIENT', how='left')
            else:
                encounter_features = encounter_patterns
            
            # Join with feature data
            feature_data = feature_data.merge(encounter_features, on='PATIENT', how='left')
        
        # Add condition features
        if self.conditions_df is not None:
            # Count conditions by patient
            condition_counts = self.conditions_df.groupby('PATIENT').size().reset_index(name='TOTAL_CONDITIONS')
            
            # Count unique conditions by patient
            unique_conditions = self.conditions_df.groupby('PATIENT')['CODE'].nunique().reset_index(name='UNIQUE_CONDITIONS')
            
            # Join condition features
            condition_features = condition_counts.merge(unique_conditions, on='PATIENT', how='left')
            
            # Join with feature data
            feature_data = feature_data.merge(condition_features, on='PATIENT', how='left')
        
        # Fill missing values
        for column in feature_data.columns:
            if pd.api.types.is_numeric_dtype(feature_data[column]):
                feature_data[column] = feature_data[column].fillna(0)
        
        # Drop non-feature columns
        drop_cols = ['NEXT_VISIT', 'DAYS_TO_NEXT_VISIT', 'THRESHOLD', 'DAYS_SINCE_LAST', 
                     'FIRST_ENCOUNTER', 'LAST_ENCOUNTER', 'START', 'STOP']
        feature_cols = [col for col in feature_data.columns if col not in drop_cols]
        
        logger.info(f"Extracted {len(feature_cols)} features for {len(feature_data)} patients")
        
        return feature_data[feature_cols]
    
    def train_deep_learning_model(self, X_train, X_test, y_train, y_test, preprocessor):
        """
        Train a deep learning model using TensorFlow for dropout prediction.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        X_test : pandas.DataFrame
            Testing features
        y_train : pandas.Series
            Training labels
        y_test : pandas.Series
            Testing labels
        preprocessor : ColumnTransformer
            Preprocessor for feature transformation
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        logger.info("Training deep learning model with TensorFlow...")
        
        # Preprocess data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get input dimensions
        input_dim = X_train_processed.shape[1]
        
        # Create a sequential model
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Define early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train_processed, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate the model
        y_pred_proba = model.predict(X_test_processed, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Log results
        logger.info("Deep learning model results:")
        logger.info(f"  - Accuracy: {accuracy:.4f}")
        logger.info(f"  - Precision: {precision:.4f}")
        logger.info(f"  - Recall: {recall:.4f}")
        logger.info(f"  - F1 Score: {f1:.4f}")
        logger.info(f"  - ROC AUC: {roc_auc:.4f}")
        logger.info(f"  - PR AUC: {pr_auc:.4f}")
        
        # Print classification report
        logger.info("Classification Report for deep learning model:")
        logger.info(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("Confusion Matrix for deep learning model:")
        logger.info(cm)
        
        # Save model visualization
        os.makedirs(os.path.join(self.output_dir, 'model_visualizations'), exist_ok=True)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_visualizations', 'deep_learning_training_history.png'))
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Deep Learning')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.output_dir, 'model_visualizations', 'confusion_matrix_deep_learning.png'))
        plt.close()
        
        # Plot PR curve
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(recall_curve, precision_curve, label=f'PR AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Deep Learning')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'model_visualizations', 'pr_curve_deep_learning.png'))
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    def train_model(self, feature_data):
        """
        Train a machine learning model to predict patient drop-offs.
        
        Parameters:
        -----------
        feature_data : pandas.DataFrame
            DataFrame with features and drop-off labels
        
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        logger.info("Training dropout prediction model...")
        
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
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
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
        models = {
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
        for name, model in models.items():
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
            
            # Store metrics
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
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
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
            
            # Plot ROC curve
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
    
    def run_pipeline(self):
        """Run the complete pipeline from data loading to model training."""
        # Load data
        self.load_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Identify drop-offs
        labeled_data = self.identify_dropoffs()
        
        if labeled_data is None:
            logger.error("Failed to identify drop-offs")
            return None
        
        # Extract features
        feature_data = self.extract_features(labeled_data)
        
        # Train model
        metrics = self.train_model(feature_data)
        
        logger.info("Pipeline complete")
        
        return metrics
