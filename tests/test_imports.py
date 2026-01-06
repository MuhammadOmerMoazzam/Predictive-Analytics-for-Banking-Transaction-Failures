"""
Basic import test to verify all modules can be imported without errors.
"""

def test_imports():
    """Test that all modules can be imported."""
    try:
        # Test model modules
        from src.model.data_processor import load_transaction_data, preprocess_data
        from src.model.model_trainer import ModelTrainer
        from src.model.model_evaluation import ModelEvaluator, ModelOptimizer
        print("SUCCESS: Model modules imported successfully")
        
        # Test simulator module
        from src.simulator.transaction_simulator import TransactionSimulator
        print("SUCCESS: Simulator module imported successfully")

        # Test API module
        from src.api.main import app
        print("SUCCESS: API module imported successfully")

        # Test integration module
        from src.integration.fineract_integration import FineractIntegration
        print("SUCCESS: Integration module imported successfully")

        # Test analytics module
        from src.analytics.reporting_module import AnalyticsReporter
        print("SUCCESS: Analytics module imported successfully")

        # Test config
        from src.api.config import API_HOST, API_PORT
        print("SUCCESS: Config imported successfully")

        print("\nAll modules imported successfully!")
        return True

    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if not success:
        exit(1)