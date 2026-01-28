"""
Backend API for the AI-Based Transaction Failure Prediction System.

This module implements a FastAPI-based API that provides transaction failure 
prediction services using the trained machine learning models.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Based Transaction Failure Prediction API",
    description="An API that predicts the probability of banking transaction failures using machine learning",
    version="1.0.0"
)

# Define request models
class TransactionRequest(BaseModel):
    """
    Request model for transaction prediction.
    """
    transaction_amount: float
    account_balance: float
    time_of_day: int
    day_of_week: int
    transaction_type: str
    location: str
    merchant_category: str
    location_risk_score: float
    historical_failure_rate: float

class TransactionBatchRequest(BaseModel):
    """
    Request model for batch transaction prediction.
    """
    transactions: List[TransactionRequest]

class PredictionResponse(BaseModel):
    """
    Response model for transaction prediction.
    """
    transaction_id: Optional[str] = None
    failure_probability: float
    prediction: int  # 0 for success, 1 for failure
    failure_reason: Optional[str] = None

# Load the trained model and scaler
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_optimized_transaction_model_random_forest.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

# Global variables for model and scaler
model = None
scaler = None
label_encoders = None

def load_model():
    """
    Load the trained model and scaler from disk.
    """
    global model, scaler, label_encoders
    
    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            # For demo purposes, we'll continue without a model
            # In a real scenario, this would be a critical error
    
        # Load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Scaler loaded from {SCALER_PATH}")
        else:
            logger.warning(f"Scaler file not found at {SCALER_PATH}")
    
        # Load label encoders if they exist
        encoders_path = "models/label_encoders.pkl"
        if os.path.exists(encoders_path):
            label_encoders = joblib.load(encoders_path)
            logger.info(f"Label encoders loaded from {encoders_path}")
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load model on startup
@app.on_event("startup")
def startup_event():
    logger.info("Loading model and scaler...")
    load_model()
    logger.info("Model and scaler loaded successfully")

@app.get("/")
def read_root():
    """
    Root endpoint to check API status.
    """
    return {
        "message": "AI-Based Transaction Failure Prediction API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_transaction_failure(transaction: TransactionRequest):
    """
    Predict the probability of a transaction failure.
    
    Args:
        transaction: Transaction details for prediction
        
    Returns:
        PredictionResponse: Prediction result with failure probability
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        # Convert transaction to DataFrame with features in the exact order used during training
        # The order is: transaction_amount, time_of_day, day_of_week, transaction_type,
        # location, account_balance, historical_failure_rate, merchant_category, location_risk_score
        transaction_df = pd.DataFrame([[
            transaction.transaction_amount,
            transaction.time_of_day,
            transaction.day_of_week,
            transaction.transaction_type,
            transaction.location,
            transaction.account_balance,
            transaction.historical_failure_rate,
            transaction.merchant_category,
            transaction.location_risk_score
        ]], columns=[
            'transaction_amount',
            'time_of_day',
            'day_of_week',
            'transaction_type',
            'location',
            'account_balance',
            'historical_failure_rate',
            'merchant_category',
            'location_risk_score'
        ])
        
        # Apply label encoding if available
        if label_encoders:
            for col in ['transaction_type', 'location', 'merchant_category']:
                if col in label_encoders and col in transaction_df.columns:
                    # Handle unseen labels by using 0 as default
                    transaction_df[col] = transaction_df[col].apply(
                        lambda x: label_encoders[col].transform([str(x)])[0] 
                        if str(x) in label_encoders[col].classes_ 
                        else 0
                    )
        
        # Scale the features
        transaction_scaled = scaler.transform(transaction_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(transaction_scaled)[0]
        failure_probability = float(prediction_proba[1])  # Probability of failure
        prediction = int(model.predict(transaction_scaled)[0])  # 0 or 1
        
        # Determine failure reason based on model confidence
        failure_reason = None
        if prediction == 1:  # If predicted as failure
            if transaction.transaction_amount > 1000:
                failure_reason = "high_amount_risk"
            elif transaction.location_risk_score > 0.8:
                failure_reason = "high_location_risk"
            elif transaction.historical_failure_rate > 0.5:
                failure_reason = "high_historical_failure"
            else:
                failure_reason = "model_prediction"
        
        return PredictionResponse(
            failure_probability=failure_probability,
            prediction=prediction,
            failure_reason=failure_reason
        )
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=List[PredictionResponse])
def predict_transaction_batch(transactions: TransactionBatchRequest):
    """
    Predict the probability of failure for multiple transactions.
    
    Args:
        transactions: List of transaction details for prediction
        
    Returns:
        List[PredictionResponse]: List of prediction results
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        # Convert transactions to DataFrame with features in the exact order used during training
        # The order is: transaction_amount, time_of_day, day_of_week, transaction_type,
        # location, account_balance, historical_failure_rate, merchant_category, location_risk_score
        transaction_rows = []
        for t in transactions.transactions:
            transaction_rows.append([
                t.transaction_amount,
                t.time_of_day,
                t.day_of_week,
                t.transaction_type,
                t.location,
                t.account_balance,
                t.historical_failure_rate,
                t.merchant_category,
                t.location_risk_score
            ])

        transaction_df = pd.DataFrame(transaction_rows, columns=[
            'transaction_amount',
            'time_of_day',
            'day_of_week',
            'transaction_type',
            'location',
            'account_balance',
            'historical_failure_rate',
            'merchant_category',
            'location_risk_score'
        ])
        
        # Apply label encoding if available
        if label_encoders:
            for col in ['transaction_type', 'location', 'merchant_category']:
                if col in label_encoders and col in transaction_df.columns:
                    transaction_df[col] = transaction_df[col].apply(
                        lambda x: label_encoders[col].transform([str(x)])[0] 
                        if str(x) in label_encoders[col].classes_ 
                        else 0
                    )
        
        # Scale the features
        transactions_scaled = scaler.transform(transaction_df)
        
        # Make predictions
        predictions_proba = model.predict_proba(transactions_scaled)
        predictions = model.predict(transactions_scaled)
        
        results = []
        for i, pred_proba in enumerate(predictions_proba):
            failure_probability = float(pred_proba[1])
            prediction = int(predictions[i])
            
            # Determine failure reason
            failure_reason = None
            if prediction == 1:  # If predicted as failure
                current_transaction = transactions.transactions[i]
                if current_transaction.transaction_amount > 1000:
                    failure_reason = "high_amount_risk"
                elif current_transaction.location_risk_score > 0.8:
                    failure_reason = "high_location_risk"
                elif current_transaction.historical_failure_rate > 0.5:
                    failure_reason = "high_historical_failure"
                else:
                    failure_reason = "model_prediction"
            
            results.append(PredictionResponse(
                failure_probability=failure_probability,
                prediction=prediction,
                failure_reason=failure_reason
            ))
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
def get_model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        return {"error": "Model not loaded"}

    # Get model type
    model_type = type(model).__name__

    # Get feature names if available (this depends on the model type)
    feature_info = {}
    if hasattr(model, 'feature_importances_'):
        feature_info['has_feature_importances'] = True
    elif hasattr(model, 'coef_'):
        feature_info['has_coefficients'] = True

    return {
        "model_type": model_type,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_count": getattr(model, 'n_features_in_', 'unknown'),
        "feature_info": feature_info
    }


# Report-related endpoints
@app.get("/reports/transaction-analysis")
def get_transaction_analysis():
    """
    Get transaction analysis report.
    """
    try:
        # For demo purposes, return sample transaction analysis
        # In a real implementation, this would load from saved reports
        return {
            "report_type": "transaction_analysis",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_transactions": 5000,
                "failed_transactions": 877,
                "success_rate": 0.8246,
                "average_transaction_amount": 150.25,
                "failure_rate_by_type": {
                    "debit": 0.18,
                    "credit": 0.15
                },
                "failure_rate_by_location": {
                    "local": 0.12,
                    "national": 0.16,
                    "international": 0.22
                }
            }
        }
    except Exception as e:
        logger.error(f"Error generating transaction analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")


@app.get("/reports/performance-metrics")
def get_performance_metrics():
    """
    Get model performance metrics report.
    """
    try:
        # For demo purposes, return sample performance metrics
        # In a real implementation, this would load from saved reports
        return {
            "report_type": "performance_metrics",
            "generated_at": datetime.now().isoformat(),
            "metrics": {
                "accuracy": 0.81,
                "precision": 0.185,
                "recall": 0.0265,
                "f1_score": 0.0463,
                "roc_auc": 0.51  # Placeholder value
            },
            "model_info": {
                "model_type": "naive_bayes",
                "training_samples": 4000,
                "testing_samples": 1000
            }
        }
    except Exception as e:
        logger.error(f"Error generating performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")


@app.get("/reports/monthly-summary/{year}/{month}")
def get_monthly_summary(year: int, month: int):
    """
    Get monthly transaction summary report.

    Args:
        year: Year for the report (e.g., 2023)
        month: Month for the report (1-12)
    """
    try:
        # Validate inputs
        if not (1 <= month <= 12):
            raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
        if year < 2000 or year > 2100:
            raise HTTPException(status_code=400, detail="Year must be between 2000 and 2100")

        # For demo purposes, return sample monthly summary
        # In a real implementation, this would load from saved reports
        return {
            "report_type": "monthly_summary",
            "year": year,
            "month": month,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_transactions": 420,
                "failed_transactions": 75,
                "success_rate": 0.8214,
                "total_amount": 63500.75,
                "average_transaction_amount": 151.19,
                "top_failure_reasons": [
                    {"reason": "high_amount_risk", "count": 32},
                    {"reason": "high_location_risk", "count": 21},
                    {"reason": "high_historical_failure", "count": 15},
                    {"reason": "model_prediction", "count": 7}
                ]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating monthly summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")


@app.get("/reports/generate-full-report")
def generate_full_report():
    """
    Generate a comprehensive analytics report.
    This would typically take some time to process.
    """
    try:
        # For demo purposes, return sample comprehensive report
        # In a real implementation, this would perform actual analytics
        return {
            "report_type": "comprehensive_analytics",
            "generated_at": datetime.now().isoformat(),
            "processing_status": "completed",
            "sections": [
                "transaction_analysis",
                "model_performance",
                "failure_patterns",
                "recommendations"
            ],
            "summary": {
                "total_transactions_analyzed": 5000,
                "report_sections": 4,
                "recommendations_count": 3
            },
            "recommendations": [
                "Consider additional verification for transactions over $1000",
                "Monitor international transactions more closely",
                "Review accounts with historical failure rate > 0.5"
            ]
        }
    except Exception as e:
        logger.error(f"Error generating full report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

# Additional endpoints for model management (for development purposes)
@app.post("/reload_model")
def reload_model():
    """
    Reload the model and scaler from disk.
    """
    try:
        load_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)