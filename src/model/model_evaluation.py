"""
Model evaluation and optimization module for the AI-Based Transaction Failure Prediction System.

This module provides tools for evaluating model performance, performing hyperparameter 
tuning, and optimizing model parameters.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve, 
                           confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from typing import Dict, List, Tuple, Any, Optional
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class to evaluate and optimize machine learning models for transaction failure prediction.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.best_params = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name: str = "Model"):
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model for reporting
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}" if roc_auc else
                   f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return results
    
    def compare_models(self, model_results: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple models based on their evaluation results.
        
        Args:
            model_results: List of dictionaries containing model evaluation results
            
        Returns:
            pd.DataFrame: DataFrame comparing model performance
        """
        comparison_data = []
        for result in model_results:
            comparison_data.append({
                'Model': result['model_name'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'ROC_AUC': result['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(by='F1_Score', ascending=False)
        
        return comparison_df
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name: str = "Model"):
        """
        Plot ROC curve for a model.
        
        Args:
            y_test: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
        """
        if y_pred_proba is None:
            logger.warning(f"Cannot plot ROC curve for {model_name} - no probability predictions available")
            return
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def plot_confusion_matrix(self, cm, model_name: str = "Model", class_names: List[str] = ['Success', 'Failure']):
        """
        Plot confusion matrix for a model.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            class_names: Names of the classes
        """
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def cross_validate_model(self, model, X, y, cv: int = 5):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Define metrics to evaluate
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            logger.info(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results

class ModelOptimizer:
    """
    Class to optimize model hyperparameters using grid search or random search.
    """
    
    def __init__(self):
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}
    
    def optimize_random_forest(self, X_train, y_train, param_distributions: Optional[Dict] = None, 
                              n_iter: int = 100, cv: int = 5, scoring: str = 'f1'):
        """
        Optimize Random Forest hyperparameters using random search.
        
        Args:
            X_train: Training features
            y_train: Training targets
            param_distributions: Parameter distributions for random search
            n_iter: Number of parameter settings sampled
            cv: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            tuple: (best_model, best_params, best_score)
        """
        if param_distributions is None:
            param_distributions = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        
        rf = RandomForestClassifier(random_state=42)
        
        logger.info("Starting Random Forest hyperparameter optimization...")
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_models['random_forest'] = random_search.best_estimator_
        self.best_params['random_forest'] = random_search.best_params_
        self.best_scores['random_forest'] = random_search.best_score_
        
        logger.info(f"Best Random Forest parameters: {random_search.best_params_}")
        logger.info(f"Best Random Forest score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
    
    def optimize_logistic_regression(self, X_train, y_train, param_grid: Optional[Dict] = None, 
                                   cv: int = 5, scoring: str = 'f1'):
        """
        Optimize Logistic Regression hyperparameters using grid search.
        
        Args:
            X_train: Training features
            y_train: Training targets
            param_grid: Parameter grid for grid search
            cv: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            tuple: (best_model, best_params, best_score)
        """
        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
        
        # Filter param_grid to only include compatible combinations
        # For l1 and elasticnet penalties, only saga solver is compatible
        valid_params = []
        for C in param_grid['C']:
            for penalty in param_grid['penalty']:
                for solver in param_grid['solver']:
                    for max_iter in param_grid['max_iter']:
                        # Check compatibility
                        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                            continue
                        if penalty == 'elasticnet' and solver != 'saga':
                            continue
                        if penalty == 'l2' and solver == 'lbfgs':
                            continue
                        valid_params.append({
                            'C': C,
                            'penalty': penalty,
                            'solver': solver,
                            'max_iter': max_iter
                        })
        
        # Create grid search with valid parameter combinations
        lr = LogisticRegression(random_state=42)
        
        logger.info("Starting Logistic Regression hyperparameter optimization...")
        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=valid_params,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_models['logistic_regression'] = grid_search.best_estimator_
        self.best_params['logistic_regression'] = grid_search.best_params_
        self.best_scores['logistic_regression'] = grid_search.best_score_
        
        logger.info(f"Best Logistic Regression parameters: {grid_search.best_params_}")
        logger.info(f"Best Logistic Regression score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def optimize_model(self, model_type: str, X_train, y_train, **kwargs):
        """
        Optimize hyperparameters for a specific model type.
        
        Args:
            model_type: Type of model to optimize ('random_forest', 'logistic_regression')
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments for optimization
            
        Returns:
            tuple: (best_model, best_params, best_score)
        """
        if model_type == 'random_forest':
            return self.optimize_random_forest(X_train, y_train, **kwargs)
        elif model_type == 'logistic_regression':
            return self.optimize_logistic_regression(X_train, y_train, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

def save_evaluation_results(results: Dict, filepath: str):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        filepath: Path to save the results
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif hasattr(value, 'tolist'):  # For numpy scalars
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {filepath}")

def load_evaluation_results(filepath: str) -> Dict:
    """
    Load evaluation results from a JSON file.
    
    Args:
        filepath: Path to the saved results
        
    Returns:
        dict: Dictionary containing evaluation results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Evaluation results loaded from {filepath}")
    return results