"""
Transaction Simulator for the AI-Based Transaction Failure Prediction System.

This module generates synthetic transaction data that mimics real banking transactions,
including both successful and failed scenarios with various failure types.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging
from typing import List, Dict, Optional
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionSimulator:
    """
    Class to simulate banking transactions with various failure scenarios.
    """
    
    def __init__(self, 
                 failure_rate: float = 0.15,
                 high_amount_threshold: float = 1000.0,
                 high_risk_locations: Optional[List[str]] = None,
                 failure_types: Optional[Dict[str, float]] = None):
        """
        Initialize the transaction simulator.
        
        Args:
            failure_rate: Overall probability of a transaction failing
            high_amount_threshold: Amount above which transactions are considered high-risk
            high_risk_locations: List of locations considered high-risk
            failure_types: Dictionary mapping failure types to their probabilities
        """
        self.failure_rate = failure_rate
        self.high_amount_threshold = high_amount_threshold
        self.high_risk_locations = high_risk_locations or ['high_risk_location_1', 'high_risk_location_2']
        self.failure_types = failure_types or {
            'timeout': 0.4,
            'network_error': 0.2,
            'insufficient_funds': 0.2,
            'duplicate_transaction': 0.1,
            'routing_error': 0.1
        }
        
        # Validate failure type probabilities sum to 1.0
        total_prob = sum(self.failure_types.values())
        if abs(total_prob - 1.0) > 1e-6:
            # Normalize probabilities
            for key in self.failure_types:
                self.failure_types[key] /= total_prob
    
    def generate_transaction_id(self) -> str:
        """Generate a unique transaction ID."""
        return str(uuid.uuid4())
    
    def generate_timestamp(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate a random timestamp between start and end dates."""
        time_between = end_date - start_date
        random_seconds = random.randint(0, int(time_between.total_seconds()))
        return start_date + timedelta(seconds=random_seconds)
    
    def generate_transaction_amount(self) -> float:
        """Generate a transaction amount following a log-normal distribution."""
        # Use log-normal distribution to simulate realistic transaction amounts
        # Most transactions are small, but some are large
        return float(np.random.lognormal(mean=3, sigma=1.5))
    
    def generate_account_info(self) -> Dict:
        """Generate account-related information."""
        balance = float(np.random.lognormal(mean=4, sigma=1))  # Higher mean for balance
        historical_failure_rate = float(np.random.beta(2, 8))  # Beta distribution for failure rate
        return {
            'account_balance': balance,
            'historical_failure_rate': historical_failure_rate
        }
    
    def generate_location(self) -> str:
        """Generate a transaction location."""
        locations = ['local', 'national', 'international'] + self.high_risk_locations
        return random.choice(locations)
    
    def generate_merchant_info(self) -> Dict:
        """Generate merchant-related information."""
        merchant_categories = ['grocery', 'gas', 'retail', 'restaurant', 'online', 'other']
        return {
            'merchant_category': random.choice(merchant_categories),
            'location_risk_score': float(np.random.uniform(0, 1))
        }
    
    def determine_failure(self, transaction_data: Dict) -> tuple:
        """
        Determine if a transaction should fail and with what type of failure.
        
        Args:
            transaction_data: Dictionary containing transaction information
            
        Returns:
            tuple: (should_fail: bool, failure_type: str or None)
        """
        # Base failure probability
        failure_prob = self.failure_rate
        
        # Increase probability based on risk factors
        if transaction_data['transaction_amount'] > self.high_amount_threshold:
            failure_prob *= 2.0  # Higher chance of failure for high-value transactions
        
        if transaction_data['location'] in self.high_risk_locations:
            failure_prob *= 1.5  # Higher chance of failure in high-risk locations
        
        if transaction_data['historical_failure_rate'] > 0.5:
            failure_prob *= 1.3  # Higher chance of failure for accounts with high historical failure rate
        
        # Cap the failure probability at 95%
        failure_prob = min(failure_prob, 0.95)
        
        # Determine if transaction should fail
        should_fail = random.random() < failure_prob
        
        if should_fail:
            # Select failure type based on configured probabilities
            failure_type = random.choices(
                list(self.failure_types.keys()),
                list(self.failure_types.values())
            )[0]
            return True, failure_type
        else:
            return False, None
    
    def generate_transaction(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Generate a single synthetic transaction.
        
        Args:
            start_date: Start date for timestamp generation
            end_date: End date for timestamp generation
            
        Returns:
            Dict: Dictionary containing transaction information
        """
        # Generate base transaction data
        transaction_id = self.generate_transaction_id()
        timestamp = self.generate_timestamp(start_date, end_date)
        transaction_amount = self.generate_transaction_amount()
        
        # Generate account information
        account_info = self.generate_account_info()
        
        # Generate location
        location = self.generate_location()
        
        # Generate merchant information
        merchant_info = self.generate_merchant_info()
        
        # Create transaction dictionary
        transaction_data = {
            'transaction_id': transaction_id,
            'timestamp': timestamp,
            'transaction_amount': transaction_amount,
            'time_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'transaction_type': random.choice(['debit', 'credit']),
            'location': location,
            **account_info,
            **merchant_info
        }
        
        # Determine if transaction should fail
        should_fail, failure_type = self.determine_failure(transaction_data)
        
        # Add failure information
        transaction_data['transaction_failure'] = int(should_fail)
        transaction_data['failure_type'] = failure_type if should_fail else 'success'
        transaction_data['failure_reason'] = failure_type if should_fail else 'none'
        
        return transaction_data
    
    def generate_transactions(self, 
                            count: int, 
                            start_date: datetime, 
                            end_date: datetime,
                            output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate multiple synthetic transactions.
        
        Args:
            count: Number of transactions to generate
            start_date: Start date for timestamp generation
            end_date: End date for timestamp generation
            output_file: Optional file path to save the generated transactions
            
        Returns:
            pd.DataFrame: DataFrame containing generated transactions
        """
        logger.info(f"Generating {count} synthetic transactions...")
        
        transactions = []
        for i in range(count):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Generated {i}/{count} transactions...")
            
            transaction = self.generate_transaction(start_date, end_date)
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Convert timestamp to string for CSV compatibility
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Generated transactions saved to {output_file}")
        
        logger.info(f"Successfully generated {len(df)} transactions "
                   f"with {df['transaction_failure'].sum()} failures "
                   f"({df['transaction_failure'].mean():.2%} failure rate)")
        
        return df

def main():
    """Main function to demonstrate the transaction simulator."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Generate synthetic banking transactions')
    parser.add_argument('--count', type=int, default=10000, 
                        help='Number of transactions to generate')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date for transactions (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='End date for transactions (YYYY-MM-DD)')
    parser.add_argument('--output-file', type=str, 
                        default='data/synthetic_transactions.csv',
                        help='Output file path for generated transactions')
    parser.add_argument('--failure-rate', type=float, default=0.15,
                        help='Overall failure rate for transactions')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Create simulator
    simulator = TransactionSimulator(failure_rate=args.failure_rate)
    
    # Generate transactions
    df = simulator.generate_transactions(
        count=args.count,
        start_date=start_date,
        end_date=end_date,
        output_file=args.output_file
    )
    
    print(f"Generated {len(df)} transactions")
    print(f"Failure rate: {df['transaction_failure'].mean():.2%}")
    print(f"Failure types distribution:\n{df['failure_type'].value_counts()}")

if __name__ == "__main__":
    main()