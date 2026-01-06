"""
Data loading and preprocessing module for the AI-Based Transaction Failure Prediction System.

This module handles loading transaction data from various sources,
cleaning the data, and preparing it for machine learning model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_transaction_data(file_path):
    """
    Load transaction data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing transaction data
        
    Returns:
        pd.DataFrame: Loaded transaction data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} transactions from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def preprocess_data(df):
    """
    Preprocess the transaction data for model training.
    
    Args:
        df (pd.DataFrame): Raw transaction data
        
    Returns:
        pd.DataFrame: Preprocessed transaction data
    """
    logger.info("Starting data preprocessing...")
    
    # Make a copy to avoid modifying the original data
    df_processed = df.copy()
    
    # Handle missing values
    # For numerical columns, use median imputation
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'transaction_failure' in numerical_cols:
        numerical_cols.remove('transaction_failure')  # Don't impute the target variable
    
    if numerical_cols:
        imputer_num = SimpleImputer(strategy='median')
        df_processed[numerical_cols] = imputer_num.fit_transform(df_processed[numerical_cols])
    
    # For categorical columns, use mode imputation
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = imputer_cat.fit_transform(df_processed[categorical_cols])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    logger.info("Data preprocessing completed")
    return df_processed, label_encoders

def prepare_features_target(df, target_column='transaction_failure'):
    """
    Separate features and target variable from the dataset.
    
    Args:
        df (pd.DataFrame): Preprocessed transaction data
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X: features, y: target)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature data
        y (pd.Series): Target data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale the features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Feature scaling completed")
    return X_train_scaled, X_test_scaled, scaler