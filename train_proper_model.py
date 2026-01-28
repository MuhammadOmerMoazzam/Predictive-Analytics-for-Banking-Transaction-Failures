"""
Script to properly train the model with only the features available at prediction time.
This resolves the feature mismatch issue.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

from src.simulator.transaction_simulator import TransactionSimulator
from src.model.data_processor import preprocess_data, prepare_features_target, split_data, scale_features
from src.model.model_trainer import ModelTrainer

def train_model_with_prediction_features():
    """Train a model using only the features that will be available at prediction time."""
    
    print("Generating synthetic transaction data...")
    simulator = TransactionSimulator(failure_rate=0.15)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate training data
    df = simulator.generate_transactions(
        count=5000,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"Generated {len(df)} transactions")
    print(f"Original columns: {list(df.columns)}")
    
    # Create a dataset with only the features that will be available at prediction time
    # These are the fields that the API expects
    prediction_features_df = df.copy()
    
    # Remove columns that won't be available at prediction time
    # (these are either targets or identifiers)
    columns_to_remove = []
    for col in prediction_features_df.columns:
        if col not in [
            'transaction_amount', 'account_balance', 'time_of_day', 'day_of_week',
            'transaction_type', 'location', 'merchant_category', 
            'location_risk_score', 'historical_failure_rate'
        ]:
            if col != 'transaction_failure':  # Keep the target for training
                columns_to_remove.append(col)
    
    print(f"Removing columns not available at prediction time: {columns_to_remove}")
    prediction_features_df = prediction_features_df.drop(columns=columns_to_remove)
    
    print(f"Final columns for training: {list(prediction_features_df.columns)}")
    
    # Now preprocess this cleaned dataset
    print("Preprocessing prediction-features dataset...")
    df_processed, label_encoders = preprocess_data(prediction_features_df)
    print(f"Processed data shape: {df_processed.shape}")
    
    # Prepare features and target
    X, y = prepare_features_target(df_processed, 'transaction_failure')
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Feature names: {list(X.columns)}")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(f"Training set shape: {X_train_scaled.shape}")
    
    # Train models
    print("Training models...")
    trainer = ModelTrainer()
    trainer.train_models(X_train_scaled, y_train)
    
    # Evaluate models
    print("Evaluating models...")
    best_model_name = trainer.evaluate_models(X_test_scaled, y_test)
    print(f"Best model: {best_model_name}")
    
    # Save the best model
    model_path = 'models/best_optimized_transaction_model_random_forest.pkl'
    trainer.save_model(best_model_name, model_path)
    print(f"Model saved to {model_path}")
    
    # Save the scaler
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save the label encoders
    encoders_path = 'models/label_encoders.pkl'
    joblib.dump(label_encoders, encoders_path)
    print(f"Label encoders saved to {encoders_path}")
    
    # Test the saved components
    print("\nTesting saved components...")
    try:
        # Load and test model
        loaded_model = joblib.load(model_path)
        print("SUCCESS: Model loaded successfully")
        print(f"Model expects {loaded_model.n_features_in_} features")
        
        # Load and test scaler
        loaded_scaler = joblib.load(scaler_path)
        print("SUCCESS: Scaler loaded successfully")
        
        # Load and test encoders
        loaded_encoders = joblib.load(encoders_path)
        print("SUCCESS: Encoders loaded successfully")
        
        # Test prediction on a sample
        sample_prediction = loaded_model.predict(X_test_scaled[:1])
        sample_proba = loaded_model.predict_proba(X_test_scaled[:1])
        print(f"SUCCESS: Sample prediction: {sample_prediction[0]}, probability: {sample_proba[0][1]:.4f}")
        
        print("\nAll components saved and tested successfully!")
        print("The API should now work correctly with the proper feature set.")
        
    except Exception as e:
        print(f"Error testing components: {e}")
        raise

if __name__ == "__main__":
    train_model_with_prediction_features()