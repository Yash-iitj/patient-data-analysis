import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_spark():
    os.makedirs('./model_results', exist_ok=True)

    # Start Spark session
    spark = SparkSession.builder \
        .appName("ModelTraining") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    # Load dataset
    print("Choose the dataset to work on")
    print("1. Small Dataset")
    print("2. Big Dataset")
    choice = input("Enter your choice here: ")
    path = './data/final_data_small.csv' if choice == '1' else './data/final_data_big.csv'

    # Load dataset WITHOUT overwriting label
    df_pd = pd.read_csv(path)
    if 'IS_DROPOFF' not in df_pd.columns:
        raise ValueError("The dataset must contain an 'IS_DROPOFF' column.")

    df = spark.createDataFrame(df_pd)

    label = 'IS_DROPOFF'
    features = [col for col in df.columns if col != label]

    # Assemble + scale
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

    # Train/test split
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Models
    models = {
        'logistic_regression': LogisticRegression(featuresCol='scaledFeatures', labelCol=label),
        'random_forest': RandomForestClassifier(featuresCol='scaledFeatures', labelCol=label, numTrees=50, maxDepth=10),
        'gradient_boosting': GBTClassifier(featuresCol='scaledFeatures', labelCol=label, maxIter=50)
    }

    evaluator = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    metrics = {}

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline(stages=[assembler, scaler, classifier])
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)

        auc = evaluator.evaluate(predictions)
        acc = predictions.filter(predictions[label] == predictions["prediction"]).count() / predictions.count()

        metrics[name] = {
            'accuracy': round(acc, 4),
            'roc_auc': round(auc, 4)
        }

        print(f"{name}: Accuracy = {acc:.4f}, AUC = {auc:.4f}")
        
        # Save model
        model_path = f'./saved_models/{name}'
        model.write().overwrite().save(model_path)
        print(f"Saved model to {model_path}")

    # Save results
    results_df = pd.DataFrame(metrics).T
    results_df.to_csv('./model_results/summary_metrics.csv')
    print("\nSaved metrics to ./model_results/summary_metrics.csv")

if __name__ == "__main__":
    train_spark()
