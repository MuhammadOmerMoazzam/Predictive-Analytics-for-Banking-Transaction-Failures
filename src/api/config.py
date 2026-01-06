"""
Configuration for the AI-Based Transaction Failure Prediction API.
"""

import os
from typing import Optional

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_LOG_LEVEL = os.getenv("API_LOG_LEVEL", "info")

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_optimized_transaction_model_random_forest.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
LABEL_ENCODERS_PATH = os.getenv("LABEL_ENCODERS_PATH", "models/label_encoders.pkl")

# Data Validation Configuration
MIN_TRANSACTION_AMOUNT = 0.01
MAX_TRANSACTION_AMOUNT = 100000.0
MIN_ACCOUNT_BALANCE = 0.0
MAX_TIME_OF_DAY = 23
MIN_TIME_OF_DAY = 0
MAX_DAY_OF_WEEK = 6
MIN_DAY_OF_WEEK = 0

# Prediction Configuration
HIGH_RISK_AMOUNT_THRESHOLD = 1000.0
HIGH_RISK_LOCATION_SCORE = 0.8
HIGH_HISTORICAL_FAILURE_RATE = 0.5

# Error Configuration
PREDICTION_ERROR_CODE = 500
VALIDATION_ERROR_CODE = 422
NOT_FOUND_ERROR_CODE = 404

# API Documentation
API_TITLE = "AI-Based Transaction Failure Prediction API"
API_DESCRIPTION = "An API that predicts the probability of banking transaction failures using machine learning"
API_VERSION = "1.0.0"