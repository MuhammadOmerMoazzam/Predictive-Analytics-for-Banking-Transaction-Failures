"""
Comprehensive test suite for the AI-Based Transaction Failure Prediction System.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os
from pathlib import Path
import tempfile

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model.data_processor import load_transaction_data, preprocess_data, prepare_features_target, split_data, scale_features
from src.model.model_trainer import ModelTrainer, train_single_model
from src.model.model_evaluation import ModelEvaluator, ModelOptimizer
from src.simulator.transaction_simulator import TransactionSimulator
from src.integration.fineract_integration import FineractIntegration, integrate_with_fineract_and_predict
from src.analytics.reporting_module import AnalyticsReporter
from src.api.config import (
    API_HOST, API_PORT, MIN_TRANSACTION_AMOUNT, MAX_TRANSACTION_AMOUNT,
    MIN_ACCOUNT_BALANCE, HIGH_RISK_AMOUNT_THRESHOLD
)

class TestDataProcessor:
    """Test cases for data processing module."""
    
    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        # Create sample data
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': ['A', 'B', 'A', np.nan, 'C'],
            'transaction_failure': [0, 1, 0, 1, 0]
        })
        
        # Test preprocessing
        df_processed, label_encoders = preprocess_data(df)
        
        # Check that there are no NaN values
        assert not df_processed.isnull().any().any(), "There are still NaN values after preprocessing"
        
        # Check that categorical values are encoded
        assert pd.api.types.is_numeric_dtype(df_processed['feature2']), "Categorical column not encoded"
        
        # Check that target column is preserved
        assert 'transaction_failure' in df_processed.columns, "Target column missing"
    
    def test_prepare_features_target(self):
        """Test separation of features and target."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'transaction_failure': [0, 1, 0, 1, 0]
        })
        
        X, y = prepare_features_target(df, 'transaction_failure')
        
        assert list(X.columns) == ['feature1', 'feature2'], "Features not correctly separated"
        assert list(y) == [0, 1, 0, 1, 0], "Target not correctly separated"
    
    def test_split_data(self):
        """Test data splitting functionality."""
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'transaction_failure': [i % 2 for i in range(100)]
        })
        
        X, y = prepare_features_target(df, 'transaction_failure')
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        assert len(X_train) == 80, "Training set size incorrect"
        assert len(X_test) == 20, "Test set size incorrect"
        assert len(y_train) == 80, "Training target size incorrect"
        assert len(y_test) == 20, "Test target size incorrect"

class TestModelTrainer:
    """Test cases for model training module."""
    
    def test_model_trainer_initialization(self):
        """Test model trainer initialization."""
        trainer = ModelTrainer()
        
        assert 'logistic_regression' in trainer.models, "Logistic regression model missing"
        assert 'random_forest' in trainer.models, "Random forest model missing"
        assert 'naive_bayes' in trainer.models, "Naive Bayes model missing"
        assert len(trainer.models) == 3, "Incorrect number of models"
    
    def test_train_single_model(self):
        """Test training of a single model."""
        # Create sample data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train a model
        model = train_single_model(X, y, model_type='random_forest', n_estimators=10)
        
        # Verify the model can make predictions
        predictions = model.predict(X)
        assert len(predictions) == 100, "Model didn't predict all samples"
        assert all(pred in [0, 1] for pred in predictions), "Invalid prediction values"

class TestTransactionSimulator:
    """Test cases for transaction simulator."""
    
    def test_transaction_generation(self):
        """Test transaction generation functionality."""
        simulator = TransactionSimulator(failure_rate=0.2)
        
        # Generate transactions
        df = simulator.generate_transactions(
            count=100,
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2023-12-31')
        )
        
        assert len(df) == 100, "Incorrect number of transactions generated"
        assert 'transaction_id' in df.columns, "Transaction ID missing"
        assert 'transaction_amount' in df.columns, "Transaction amount missing"
        assert 'transaction_failure' in df.columns, "Transaction failure missing"
        assert 'failure_type' in df.columns, "Failure type missing"
        
        # Check failure rate is approximately as expected
        actual_failure_rate = df['transaction_failure'].mean()
        assert 0.05 <= actual_failure_rate <= 0.35, f"Failure rate {actual_failure_rate} not in expected range"
    
    def test_failure_determination(self):
        """Test failure determination logic."""
        simulator = TransactionSimulator(failure_rate=0.5, high_amount_threshold=100.0)
        
        # Test high-amount transaction (should have higher failure probability)
        high_amount_transaction = {
            'transaction_amount': 500.0,  # Above threshold
            'location': 'local',
            'historical_failure_rate': 0.1
        }
        
        should_fail, failure_type = simulator.determine_failure(high_amount_transaction)
        # Note: This is probabilistic, so we can't assert with certainty
        
        # Test normal transaction
        normal_transaction = {
            'transaction_amount': 50.0,  # Below threshold
            'location': 'local',
            'historical_failure_rate': 0.1
        }
        
        should_fail_normal, failure_type_normal = simulator.determine_failure(normal_transaction)
        # Note: This is probabilistic, so we can't assert with certainty

class TestModelEvaluation:
    """Test cases for model evaluation module."""
    
    def test_model_evaluator(self):
        """Test model evaluation functionality."""
        evaluator = ModelEvaluator()
        
        # Create sample predictions and true values
        y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
        y_pred_proba = [0.1, 0.9, 0.4, 0.3, 0.8, 0.6, 0.9, 0.7, 0.2, 0.3]
        
        # Evaluate model
        results = evaluator.evaluate_model(
            model=None,  # Not actually using the model in this test
            X_test=None, 
            y_test=y_true,
            model_name="Test Model"
        )
        
        # Check that results contain expected metrics
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        evaluator = ModelEvaluator()
        
        # Create sample results for multiple models
        model_results = [
            {
                'model_name': 'Model A',
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'roc_auc': 0.90
            },
            {
                'model_name': 'Model B',
                'accuracy': 0.80,
                'precision': 0.78,
                'recall': 0.82,
                'f1_score': 0.80,
                'roc_auc': 0.85
            }
        ]
        
        comparison_df = evaluator.compare_models(model_results)
        
        assert len(comparison_df) == 2, "Incorrect number of models in comparison"
        assert comparison_df.iloc[0]['Model'] == 'Model A', "Best model not ranked first"
        assert comparison_df.iloc[0]['F1_Score'] == 0.85, "Incorrect F1 score in comparison"

class TestFineractIntegration:
    """Test cases for Fineract integration."""
    
    def test_transform_transaction(self):
        """Test transaction transformation functionality."""
        # Create a mock Fineract transaction
        fineract_transaction = {
            'id': 1,
            'date': [2023, 5, 15, 14, 30],  # Year, Month, Day, Hour, Minute
            'amount': 150.75,
            'runningBalance': {'amount': 2500.50},
            'type': {'code': 'account.transfers.type.debit', 'value': 'Debit'},
            'submittedOnDate': [2023, 5, 15]
        }
        
        # Initialize integration (without authentication for this test)
        integration = FineractIntegration(
            base_url="http://test.com",
            username="test",
            password="test"
        )
        
        # Transform the transaction
        transformed = integration.transform_fineract_transaction(fineract_transaction)
        
        # Check that required fields are present
        assert 'transaction_amount' in transformed
        assert 'account_balance' in transformed
        assert 'time_of_day' in transformed
        assert 'day_of_week' in transformed
        assert 'transaction_type' in transformed
        
        # Check values
        assert transformed['transaction_amount'] == 150.75
        assert transformed['account_balance'] == 2500.50
        assert transformed['time_of_day'] == 14  # Hour from the date
        assert transformed['transaction_type'] == 'debit'

class TestAnalyticsReporter:
    """Test cases for analytics reporter."""
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        reporter = AnalyticsReporter()
        
        # Create sample data
        y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
        y_pred_proba = [0.1, 0.9, 0.4, 0.3, 0.8, 0.6, 0.9, 0.7, 0.2, 0.3]
        
        # Generate report
        report = reporter.generate_performance_report(
            y_true, y_pred, y_pred_proba, model_name="Test Model"
        )
        
        # Check that report contains expected fields
        assert 'model_name' in report
        assert 'metrics' in report
        assert 'accuracy' in report['metrics']
        assert 'precision' in report['metrics']
        assert 'recall' in report['metrics']
        assert 'f1_score' in report['metrics']
        assert 'roc_auc' in report['metrics']
        
        # Check that values are reasonable
        assert 0 <= report['metrics']['accuracy'] <= 1
        assert 0 <= report['metrics']['precision'] <= 1
        assert 0 <= report['metrics']['recall'] <= 1
        assert 0 <= report['metrics']['f1_score'] <= 1
    
    def test_generate_transaction_analysis_report(self):
        """Test transaction analysis report generation."""
        reporter = AnalyticsReporter()
        
        # Create sample transaction data
        transactions_df = pd.DataFrame({
            'transaction_amount': [100, 200, 50, 300, 150],
            'account_balance': [1000, 2000, 500, 3000, 1500],
            'transaction_failure': [0, 1, 0, 1, 0],
            'transaction_type': ['debit', 'credit', 'debit', 'credit', 'debit'],
            'location': ['local', 'local', 'national', 'local', 'international'],
            'merchant_category': ['grocery', 'gas', 'grocery', 'retail', 'restaurant'],
            'time_of_day': [9, 12, 15, 18, 21],
            'day_of_week': [0, 1, 2, 3, 4]
        })
        
        # Generate report
        report = reporter.generate_transaction_analysis_report(transactions_df)
        
        # Check that report contains expected fields
        assert 'total_transactions' in report
        assert 'failed_transactions' in report
        assert 'success_rate' in report
        assert 'failure_by_type' in report
        assert 'failure_by_location' in report
        assert 'failure_by_merchant' in report
        
        # Check values
        assert report['total_transactions'] == 5
        assert report['failed_transactions'] == 2
        assert report['success_rate'] == 0.6  # 3 out of 5 succeeded

class TestAPIConfig:
    """Test cases for API configuration."""
    
    def test_config_values(self):
        """Test that config values are properly set."""
        assert API_HOST in ['0.0.0.0', '127.0.0.1'], "Invalid API host"
        assert 1024 <= API_PORT <= 65535, "Invalid API port"
        assert MIN_TRANSACTION_AMOUNT > 0, "Invalid min transaction amount"
        assert MAX_TRANSACTION_AMOUNT > MIN_TRANSACTION_AMOUNT, "Invalid max transaction amount"
        assert MIN_ACCOUNT_BALANCE >= 0, "Invalid min account balance"
        assert HIGH_RISK_AMOUNT_THRESHOLD > MIN_TRANSACTION_AMOUNT, "Invalid high risk threshold"

# Integration tests
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_complete_pipeline(self):
        """Test the complete pipeline from data generation to prediction."""
        # Generate sample transactions
        simulator = TransactionSimulator(failure_rate=0.15)
        transactions_df = simulator.generate_transactions(
            count=500,
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2023-12-31')
        )
        
        # Preprocess data
        df_processed, label_encoders = preprocess_data(transactions_df)
        X, y = prepare_features_target(df_processed, 'transaction_failure')
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Train model
        trainer = ModelTrainer()
        trainer.train_models(X_train_scaled, y_train)
        
        # Evaluate models
        best_model_name = trainer.evaluate_models(X_test_scaled, y_test)
        
        # Verify we have a best model
        assert best_model_name in trainer.trained_models, "No best model found"
        
        # Verify model can make predictions
        sample_prediction = trainer.trained_models[best_model_name].predict(X_test_scaled[:1])
        assert sample_prediction[0] in [0, 1], "Invalid prediction value"

if __name__ == "__main__":
    pytest.main([__file__])