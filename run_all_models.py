#!/usr/bin/env python3
"""
Simplified script for Hospital Records Mining for Patient Drop-off Prediction.
This script runs all models automatically without requiring command line arguments.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set fixed parameters
class Args:
    big_data_dir = './data/data'
    test_size = 0.2
    random_state = 424
    output_dir = 'images'
    use_deep_learning = True
    scale_factor = 100

def main():
    """Main execution function."""
    # Create necessary directories
    os.makedirs('final/images', exist_ok=True)
    
    args = Args()
    logger.info("Starting pipeline with fixed parameters")
    
    try:
        # Run big data pipeline with deep learning
        logger.info("Running big data pipeline with deep learning")
        results = run_big_data_pipeline(args)
        
        # Generate comparison plots
        logger.info("Generating comparison plots")
        generate_comparison_plots(results, args.output_dir)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

def run_big_data_pipeline(args):
    """Run the pipeline using the big data with deep learning."""
    logger.info(f"Running big data pipeline with deep learning")
    
    # Check if big data exists
    if not os.path.exists(args.big_data_dir):
        logger.error(f"Big data directory not found at {args.big_data_dir}")
        raise FileNotFoundError(f"Big data directory not found at {args.big_data_dir}")
    
    # Import enhanced big data processor
    from enhanced_processor import EnhancedBigDataProcessor
    
    # Create processor
    processor = EnhancedBigDataProcessor(
        data_dir=args.big_data_dir,
        output_dir=args.output_dir,
        random_state=args.random_state,
        use_deep_learning=args.use_deep_learning
    )
    
    # Run pipeline
    results = processor.run_pipeline()
    
    # Log results
    logger.info(f"Big data pipeline complete. Results:")
    for key, value in results.items():
        logger.info(f"  - {key}: {value:.4f}")
    
    return processor.get_detailed_results()

def generate_comparison_plots(results, output_dir):
    """Generate comparison plots for model performance."""
    # Extract model names and metrics
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    
    # Create a dataframe for plotting
    plot_data = []
    for model_name in model_names:
        model_results = results[model_name]
        for metric in metrics:
            plot_data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': model_results[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Plot overall comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Value', hue='Metric', data=plot_df)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    # Plot individual metrics
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        metric_data = plot_df[plot_df['Metric'] == metric]
        sns.barplot(x='Model', y='Value', data=metric_data)
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.xlabel('Model')
        plt.ylabel(f'{metric.replace("_", " ").title()} Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
    
    # Plot confusion matrices
    for model_name in model_names:
        if 'confusion_matrix' in results[model_name] and results[model_name]['confusion_matrix'] is not None:
            cm = results[model_name]['confusion_matrix']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['No Dropoff', 'Dropoff'],
                        yticklabels=['No Dropoff', 'Dropoff'])
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
            plt.close()
    
    # Plot feature importance if available
    for model_name in model_names:
        if 'feature_importance' in results[model_name]:
            feature_imp = results[model_name]['feature_importance']
            if feature_imp is not None and len(feature_imp) > 0:
                # Sort by importance
                sorted_idx = np.argsort(feature_imp['importance'])
                
                # Plot top 15 features
                plt.figure(figsize=(10, 8))
                plt.barh(range(min(15, len(sorted_idx))), 
                        [feature_imp['importance'][i] for i in sorted_idx[-15:]], 
                        align='center')
                plt.yticks(range(min(15, len(sorted_idx))), 
                          [feature_imp['feature'][i] for i in sorted_idx[-15:]])
                plt.title(f'Feature Importance - {model_name}')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name}.png'))
                plt.close()

def print_final_results():
    """Print a summary of the results at the end."""
    print("\n" + "="*80)
    print("PATIENT DROP-OFF PREDICTION - FINAL RESULTS")
    print("="*80)
    
    # Load the most recent results
    results_files = [f for f in os.listdir('logs') if f.startswith('run_')]
    if not results_files:
        print("No results found.")
        return
    
    latest_log = max(results_files)
    with open(os.path.join('logs', latest_log), 'r') as f:
        log_content = f.read()
    
    # Extract and print model performance
    print("\nMODEL PERFORMANCE:")
    print("-"*80)
    
    models = ['logistic_regression', 'random_forest', 'gradient_boosting', 'deep_learning']
    for model in models:
        if model in log_content:
            print(f"\n{model.replace('_', ' ').title()}:")
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC']
            for metric in metrics:
                import re
                pattern = rf"{model} results:.*?- {metric}: (0\.\d+)"
                match = re.search(pattern, log_content, re.DOTALL)
                if match:
                    print(f"  - {metric}: {match.group(1)}")
    
    # Print best model
    best_model_match = re.search(r"Best model: (\w+) with ROC AUC = (0\.\d+)", log_content)
    if best_model_match:
        print("\nBEST MODEL:")
        print("-"*80)
        print(f"{best_model_match.group(1).replace('_', ' ').title()} (ROC AUC: {best_model_match.group(2)})")
    
    print("\nVisualization images saved to: final/images/")
    print("="*80)

if __name__ == "__main__":
    main()
    print_final_results()
