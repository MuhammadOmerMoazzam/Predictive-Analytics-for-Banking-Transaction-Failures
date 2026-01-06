"""
Machine learning model training module for the AI-Based Transaction Failure Prediction System.

This module implements various ML algorithms for predicting transaction failures,
including Logistic Regression, Random Forest, and Naive Bayes.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class to handle training of different ML models for transaction failure prediction.
    """
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'naive_bayes': GaussianNB()
        }
        self.trained_models = {}
        self.model_scores = {}
    
    def train_models(self, X_train, y_train):
        """
        Train all available models on the training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info("Starting model training...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            logger.info(f"Completed training {name}")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on the test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        logger.info("Starting model evaluation...")
        
        for name, model in self.trained_models.items():
            logger.info(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store results
            self.model_scores[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                       f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Find best model
        best_model_name = max(self.model_scores.keys(), 
                             key=lambda x: self.model_scores[x]['f1_score'])
        logger.info(f"Best model based on F1 score: {best_model_name}")
        
        return best_model_name
    
    def get_model_scores(self):
        """
        Get the evaluation scores for all models.
        
        Returns:
            dict: Dictionary containing scores for each model
        """
        return self.model_scores
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path where the model should be saved
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        joblib.dump(self.trained_models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name, filepath):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path from where the model should be loaded
        """
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        logger.info(f"Model {model_name} loaded from {filepath}")

def train_single_model(X_train, y_train, model_type='random_forest', **kwargs):
    """
    Train a single model of the specified type.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type (str): Type of model to train ('logistic_regression', 'random_forest', 'naive_bayes')
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Trained model
    """
    if model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_estimators=100, **kwargs)
    elif model_type == 'naive_bayes':
        model = GaussianNB(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    logger.info(f"Completed training {model_type} model")
    
    return model