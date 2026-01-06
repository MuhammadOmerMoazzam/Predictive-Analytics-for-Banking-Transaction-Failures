"""
Apache Fineract Integration Module for the AI-Based Transaction Failure Prediction System.

This module provides functionality to interact with Apache Fineract, an open-source
core banking platform, to retrieve transaction data and apply AI-based failure predictions.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineractIntegration:
    """
    Class to handle integration with Apache Fineract.
    """
    
    def __init__(self, base_url: str, username: str, password: str, tenant_id: str = "default"):
        """
        Initialize the Fineract integration.
        
        Args:
            base_url: Base URL of the Fineract instance
            username: Fineract username
            password: Fineract password
            tenant_id: Fineract tenant ID
        """
        self.base_url = base_url.rstrip('/') + '/'
        self.username = username
        self.password = password
        self.tenant_id = tenant_id
        self.auth_token = None
        
        # API endpoints
        self.endpoints = {
            'authenticate': 'authentication',
            'accounts': 'accounts',
            'transactions': 'accounts/{account_id}/transactions',
            'clients': 'clients',
            'loans': 'loans'
        }
    
    def authenticate(self) -> bool:
        """
        Authenticate with the Fineract instance to get an authentication token.
        
        Returns:
            bool: True if authentication is successful, False otherwise
        """
        try:
            url = urljoin(self.base_url, self.endpoints['authenticate'])
            
            # Prepare authentication payload
            auth_data = {
                'username': self.username,
                'password': self.password
            }
            
            # Make authentication request
            response = requests.post(
                url,
                json=auth_data,
                headers={
                    'Content-Type': 'application/json',
                    'Fineract-Platform-TenantId': self.tenant_id
                }
            )
            
            if response.status_code == 200:
                auth_result = response.json()
                self.auth_token = auth_result.get('base64EncodedAuthenticationKey')
                
                if self.auth_token:
                    logger.info("Successfully authenticated with Fineract")
                    return True
                else:
                    logger.error("Authentication successful but no token returned")
                    return False
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            return False
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a request to the Fineract API.
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method (GET, POST, PUT, DELETE)
            data: Data to send with the request (for POST/PUT)
            
        Returns:
            dict: Response data or None if request failed
        """
        if not self.auth_token:
            logger.error("Not authenticated with Fineract")
            return None
        
        try:
            url = urljoin(self.base_url, endpoint)
            
            headers = {
                'Authorization': f'Basic {self.auth_token}',
                'Fineract-Platform-TenantId': self.tenant_id,
                'Content-Type': 'application/json'
            }
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            if response.status_code in [200, 201]:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Error making request to Fineract: {str(e)}")
            return None
    
    def get_client_transactions(self, client_id: int, 
                              from_date: Optional[str] = None, 
                              to_date: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get transactions for a specific client.
        
        Args:
            client_id: ID of the client
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            List of transaction dictionaries
        """
        # First get the client's accounts
        accounts = self._make_request(f"clients/{client_id}/accounts")
        if not accounts:
            logger.error(f"Could not retrieve accounts for client {client_id}")
            return None
        
        all_transactions = []
        
        # Get transactions for each account
        for account in accounts.get('accountSummary', []):
            account_id = account.get('id')
            if account_id:
                # Construct endpoint for account transactions
                endpoint = f"accounts/{account_id}/transactions"
                
                # Add date filters if provided
                params = []
                if from_date:
                    params.append(f"fromDate={from_date}")
                if to_date:
                    params.append(f"toDate={to_date}")
                
                if params:
                    endpoint += "?" + "&".join(params)
                
                transactions = self._make_request(endpoint)
                if transactions:
                    # Add account and client info to each transaction
                    for transaction in transactions.get('pageItems', []):
                        transaction['client_id'] = client_id
                        transaction['account_id'] = account_id
                        all_transactions.append(transaction)
        
        logger.info(f"Retrieved {len(all_transactions)} transactions for client {client_id}")
        return all_transactions
    
    def get_loan_transactions(self, loan_id: int) -> Optional[List[Dict]]:
        """
        Get transactions for a specific loan.
        
        Args:
            loan_id: ID of the loan
            
        Returns:
            List of transaction dictionaries
        """
        endpoint = f"loans/{loan_id}/transactions"
        transactions = self._make_request(endpoint)
        
        if transactions:
            logger.info(f"Retrieved {len(transactions.get('pageItems', []))} transactions for loan {loan_id}")
            return transactions.get('pageItems', [])
        
        return None
    
    def get_recent_transactions(self, days: int = 7) -> Optional[List[Dict]]:
        """
        Get transactions from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of transaction dictionaries
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        from_date_str = start_date.strftime('%Y-%m-%d')
        to_date_str = end_date.strftime('%Y-%m-%d')
        
        # For this implementation, we'll get all transactions in a simple way
        # In a real implementation, you would need to iterate through accounts
        # or use more specific endpoints
        endpoint = f"accounts/transactions"
        
        params = f"?fromDate={from_date_str}&toDate={to_date_str}"
        endpoint += params
        
        transactions = self._make_request(endpoint)
        
        if transactions:
            logger.info(f"Retrieved {len(transactions.get('pageItems', []))} recent transactions")
            return transactions.get('pageItems', [])
        
        return None
    
    def transform_fineract_transaction(self, transaction: Dict) -> Dict:
        """
        Transform a Fineract transaction into the format expected by our prediction model.
        
        Args:
            transaction: Transaction data from Fineract
            
        Returns:
            Dict: Transformed transaction in model input format
        """
        # Extract relevant fields from Fineract transaction
        # This is a simplified transformation - in reality, you'd map Fineract fields
        # to your model's expected features
        transformed = {
            'transaction_amount': float(transaction.get('amount', 0)),
            'account_balance': float(transaction.get('runningBalance', {}).get('amount', 0)),
            'time_of_day': int(transaction.get('date', [datetime.now().year, datetime.now().month, datetime.now().day])[3] if len(transaction.get('date', [])) > 3 else 12),  # Default to noon
            'day_of_week': datetime(
                year=transaction.get('date', [datetime.now().year])[0],
                month=transaction.get('date', [datetime.now().month])[1],
                day=transaction.get('date', [datetime.now().day])[2]
            ).weekday(),
            'transaction_type': 'debit' if transaction.get('type', {}).get('code', '').endswith('debit') else 'credit',
            'location': 'local',  # Fineract might not have location data
            'merchant_category': 'other',  # Default category
            'location_risk_score': 0.1,  # Default low risk
            'historical_failure_rate': 0.05  # Default low failure rate
        }
        
        # Add any additional fields from the original transaction
        transformed['fineract_transaction_id'] = transaction.get('id')
        transformed['fineract_transaction_type'] = transaction.get('type', {}).get('value', 'Unknown')
        transformed['fineract_transaction_date'] = '-'.join(map(str, transaction.get('date', [])))
        
        return transformed

def integrate_with_fineract_and_predict(
    fineract_integration: FineractIntegration,
    prediction_api_url: str,
    client_id: int,
    days: int = 7
) -> List[Dict]:
    """
    Integrate with Fineract to get transactions and predict failures.
    
    Args:
        fineract_integration: Initialized FineractIntegration instance
        prediction_api_url: URL of the prediction API
        client_id: ID of the client to get transactions for
        days: Number of days to look back for transactions
        
    Returns:
        List of transactions with prediction results
    """
    # Get transactions from Fineract
    transactions = fineract_integration.get_client_transactions(
        client_id=client_id,
        from_date=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        to_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    if not transactions:
        logger.warning(f"No transactions found for client {client_id}")
        return []
    
    results = []
    
    # Transform each transaction and make predictions
    for transaction in transactions:
        # Transform Fineract transaction to model input format
        transformed_transaction = fineract_integration.transform_fineract_transaction(transaction)
        
        # Make prediction using the API
        try:
            response = requests.post(
                f"{prediction_api_url}/predict",
                json=transformed_transaction,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                prediction_result = response.json()
                result = {
                    'fineract_transaction': transaction,
                    'model_input': transformed_transaction,
                    'prediction': prediction_result
                }
                results.append(result)
            else:
                logger.error(f"Prediction API error: {response.status_code} - {response.text}")
                # Still add the transaction with an error marker
                result = {
                    'fineract_transaction': transaction,
                    'model_input': transformed_transaction,
                    'prediction': {'error': f'API error: {response.status_code}'}
                }
                results.append(result)
        
        except Exception as e:
            logger.error(f"Error making prediction request: {str(e)}")
            result = {
                'fineract_transaction': transaction,
                'model_input': transformed_transaction,
                'prediction': {'error': f'Request error: {str(e)}'}
            }
            results.append(result)
    
    logger.info(f"Processed {len(results)} transactions with predictions")
    return results

def main():
    """Main function to demonstrate Fineract integration."""
    import os
    
    # Get configuration from environment variables
    fineract_url = os.getenv('FINERACT_URL', 'http://localhost:8080/fineract-provider/api/v1/')
    fineract_username = os.getenv('FINERACT_USERNAME', 'mifos')
    fineract_password = os.getenv('FINERACT_PASSWORD', 'password')
    fineract_tenant = os.getenv('FINERACT_TENANT', 'default')
    prediction_api_url = os.getenv('PREDICTION_API_URL', 'http://localhost:8000')
    client_id = int(os.getenv('CLIENT_ID', '1'))
    
    # Initialize Fineract integration
    fineract = FineractIntegration(
        base_url=fineract_url,
        username=fineract_username,
        password=fineract_password,
        tenant_id=fineract_tenant
    )
    
    # Authenticate with Fineract
    if not fineract.authenticate():
        logger.error("Failed to authenticate with Fineract")
        return
    
    # Get transactions and make predictions
    results = integrate_with_fineract_and_predict(
        fineract_integration=fineract,
        prediction_api_url=prediction_api_url,
        client_id=client_id,
        days=7
    )
    
    # Print summary
    if results:
        prediction_count = len(results)
        failure_predictions = sum(1 for r in results if r['prediction'].get('prediction') == 1)
        
        print(f"Processed {prediction_count} transactions")
        print(f"Predicted {failure_predictions} failures ({failure_predictions/prediction_count*100:.2f}%)")
        
        # Print first few results
        for i, result in enumerate(results[:3]):
            print(f"\nTransaction {i+1}:")
            print(f"  Amount: {result['model_input']['transaction_amount']}")
            print(f"  Prediction: {result['prediction']['prediction']}")
            print(f"  Failure Probability: {result['prediction']['failure_probability']:.4f}")
    else:
        print("No transactions found or processed")

if __name__ == "__main__":
    main()