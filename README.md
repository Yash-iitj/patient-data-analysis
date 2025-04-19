# Patient Drop-off Analysis

This folder contains all the necessary code and data to run a comprehensive patient drop-off analysis using the Synthea healthcare dataset.

## Overview

The analysis predicts which patients are likely to drop off from healthcare services based on various features extracted from the Synthea dataset. It trains and evaluates multiple machine learning models, including:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Deep Learning (TensorFlow)

## Folder Structure

- `run_all_models.py`: Main script that runs all models without requiring command-line arguments
- `enhanced_processor.py`: Extended BigDataProcessor class with additional functionality for detailed results
- `src/`: Contains the original implementations:
  - `big_data_processor.py`: Core implementation for processing healthcare data
  - `big_data_generator.py`: Script for generating synthetic healthcare data
- `data/`: Contains the Synthea healthcare dataset
- `images/`: Output directory for model visualizations and comparison plots
- `logs/`: Directory for log files

## Running the Analysis

To run the patient drop-off analysis, simply execute:

```bash
python run_all_models.py
```

This will:

1. Load and preprocess the Synthea dataset
2. Identify patient drop-offs
3. Extract features for prediction
4. Train multiple machine learning models
5. Evaluate model performance
6. Generate visualizations
7. Save results to the `images/` directory

## Results

The analysis will output:

- Model performance metrics (accuracy, precision, recall, F1 score, ROC AUC, PR AUC)
- Confusion matrices
- Feature importance plots
- Model comparison visualizations

The best model will be identified based on ROC AUC score.

## Dataset

The analysis uses the Synthea synthetic healthcare dataset with:

- 10,600+ patients
- 53,000+ encounters
- 106,000+ conditions

## Generating Additional Data

If you need to generate additional synthetic healthcare data, you can use the included data generator script:

```bash
python src/big_data_generator.py --output_dir data/new_data --num_patients 1000 --scale_factor 100
```

This will create a new set of synthetic data with the specified number of patients and scale factor.

## Dependencies

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tensorflow
- xgboost
