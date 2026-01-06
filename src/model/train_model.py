"""
Main training script for the AI-Based Transaction Failure Prediction System.

This script orchestrates the entire model training process:
1. Load and preprocess transaction data
2. Split data into training and testing sets
3. Train multiple ML models
4. Evaluate model performance
5. Save the best performing model
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.model.data_processor import load_transaction_data, preprocess_data, prepare_features_target, split_data, scale_features
from src.model.model_trainer import ModelTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(data_path, target_column='transaction_failure', model_output_path=None):
    """
    Main function to train transaction failure prediction models.
    
    Args:
        data_path (str): Path to the input transaction data file
        target_column (str): Name of the target column to predict
        model_output_path (str): Path where the best model should be saved
    """
    logger.info("Starting model training process...")
    
    # Step 1: Load data
    logger.info("Step 1: Loading transaction data...")
    df = load_transaction_data(data_path)
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data...")
    df_processed, label_encoders = preprocess_data(df)
    
    # Step 3: Prepare features and target
    logger.info("Step 3: Separating features and target...")
    X, y = prepare_features_target(df_processed, target_column)
    
    # Step 4: Split data
    logger.info("Step 4: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Scale features
    logger.info("Step 5: Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 6: Train models
    logger.info("Step 6: Training models...")
    trainer = ModelTrainer()
    trainer.train_models(X_train_scaled, y_train)
    
    # Step 7: Evaluate models
    logger.info("Step 7: Evaluating models...")
    best_model_name = trainer.evaluate_models(X_test_scaled, y_test)
    
    # Step 8: Save the best model
    if model_output_path:
        logger.info("Step 8: Saving the best model...")
        trainer.save_model(best_model_name, model_output_path)
    
    logger.info(f"Training process completed. Best model: {best_model_name}")
    
    # Print model comparison summary
    logger.info("\nModel Comparison Summary:")
    scores = trainer.get_model_scores()
    for model_name, metrics in scores.items():
        status = " <- BEST" if model_name == best_model_name else ""
        logger.info(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}{status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train transaction failure prediction models')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the input transaction data file')
    parser.add_argument('--target_column', type=str, default='transaction_failure',
                        help='Name of the target column to predict')
    parser.add_argument('--model_output_path', type=str,
                        help='Path where the best model should be saved')
    
    args = parser.parse_args()
    
    # If no model output path is specified, create a default one
    if not args.model_output_path:
        args.model_output_path = os.path.join(project_root, 'models', 'best_transaction_model.pkl')
        os.makedirs(os.path.dirname(args.model_output_path), exist_ok=True)
    
    main(args.data_path, args.target_column, args.model_output_path)