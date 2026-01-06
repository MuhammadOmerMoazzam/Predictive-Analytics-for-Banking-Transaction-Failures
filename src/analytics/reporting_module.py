"""
Reporting and Analytics Module for the AI-Based Transaction Failure Prediction System.

This module provides functionality to generate reports and visualizations for
transaction failure analysis and prediction performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsReporter:
    """
    Class to generate reports and analytics for transaction failure predictions.
    """
    
    def __init__(self):
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
    
    def generate_performance_report(self, 
                                 y_true: List[int], 
                                 y_pred: List[int], 
                                 y_pred_proba: Optional[List[float]] = None,
                                 model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate a performance report for the model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            dict: Performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate ROC AUC if probabilities are provided
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                logger.warning("Could not calculate ROC AUC - only one class present in y_true")

        # Format ROC AUC for the summary
        roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"

        # Create report
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'summary': f"""
            Performance Report for {model_name}
            ==================================
            Accuracy:  {accuracy:.4f}
            Precision: {precision:.4f}
            Recall:    {recall:.4f}
            F1 Score:  {f1:.4f}
            ROC AUC:   {roc_auc_str}
            """
        }
        
        logger.info(f"Performance report generated for {model_name}")
        return report
    
    def generate_transaction_analysis_report(self, 
                                          transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate an analysis report for transaction data.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            dict: Analysis report
        """
        # Basic statistics
        total_transactions = len(transactions_df)
        failed_transactions = transactions_df['transaction_failure'].sum()
        success_rate = 1 - (failed_transactions / total_transactions)
        
        # Amount statistics
        avg_transaction_amount = transactions_df['transaction_amount'].mean()
        median_transaction_amount = transactions_df['transaction_amount'].median()
        
        # Failure analysis
        failure_by_type = transactions_df.groupby('transaction_type')['transaction_failure'].mean()
        failure_by_location = transactions_df.groupby('location')['transaction_failure'].mean()
        failure_by_merchant = transactions_df.groupby('merchant_category')['transaction_failure'].mean()
        
        # Time-based analysis
        failure_by_time_of_day = transactions_df.groupby('time_of_day')['transaction_failure'].mean()
        failure_by_day_of_week = transactions_df.groupby('day_of_week')['transaction_failure'].mean()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_transactions': total_transactions,
            'failed_transactions': int(failed_transactions),
            'success_rate': success_rate,
            'failure_rate': 1 - success_rate,
            'avg_transaction_amount': avg_transaction_amount,
            'median_transaction_amount': median_transaction_amount,
            'failure_by_type': failure_by_type.to_dict(),
            'failure_by_location': failure_by_location.to_dict(),
            'failure_by_merchant': failure_by_merchant.to_dict(),
            'failure_by_time_of_day': failure_by_time_of_day.to_dict(),
            'failure_by_day_of_week': failure_by_day_of_week.to_dict(),
            'summary': f"""
            Transaction Analysis Report
            =========================
            Total Transactions: {total_transactions}
            Failed Transactions: {int(failed_transactions)}
            Success Rate: {success_rate:.2%}
            Average Transaction Amount: ${avg_transaction_amount:.2f}
            Median Transaction Amount: ${median_transaction_amount:.2f}
            """
        }
        
        logger.info("Transaction analysis report generated")
        return report
    
    def create_prediction_visualizations(self, 
                                       y_true: List[int], 
                                       y_pred: List[int], 
                                       y_pred_proba: Optional[List[float]] = None,
                                       save_path: Optional[str] = None) -> None:
        """
        Create visualizations for model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the visualization (optional)
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Prediction Analysis', fontsize=16)
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Success', 'Failure'], 
                   yticklabels=['Success', 'Failure'], ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # Prediction Distribution
        axes[0,1].hist([np.array(y_pred)[np.array(y_true) == 0], 
                        np.array(y_pred)[np.array(y_true) == 1]], 
                       bins=2, label=['Actual Success', 'Actual Failure'], 
                       alpha=0.7, color=['green', 'red'])
        axes[0,1].set_title('Prediction Distribution by Actual Outcome')
        axes[0,1].set_xlabel('Prediction (0=Success, 1=Failure)')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend()
        
        # If probabilities are available, create probability distribution
        if y_pred_proba is not None:
            axes[1,0].hist([np.array(y_pred_proba)[np.array(y_true) == 0], 
                            np.array(y_pred_proba)[np.array(y_true) == 1]], 
                           bins=20, label=['Actual Success', 'Actual Failure'], 
                           alpha=0.7, color=['green', 'red'])
            axes[1,0].set_title('Prediction Probability Distribution')
            axes[1,0].set_xlabel('Failure Probability')
            axes[1,0].set_ylabel('Count')
            axes[1,0].legend()
        else:
            axes[1,0].text(0.5, 0.5, 'Probabilities not available', 
                          horizontalalignment='center', verticalalignment='center', 
                          transform=axes[1,0].transAxes)
            axes[1,0].set_title('Prediction Probability Distribution')
        
        # Feature importance if available (mock data for now)
        feature_names = ['Amount', 'Balance', 'Time of Day', 'Day of Week', 'Type', 'Location', 'Merchant', 'Risk Score', 'Hist Failure']
        feature_importance = np.random.rand(len(feature_names))  # Mock data
        axes[1,1].bar(feature_names, feature_importance)
        axes[1,1].set_title('Feature Importance (Mock Data)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualizations saved to {save_path}")
        
        plt.show()
    
    def create_transaction_analysis_visualizations(self, 
                                                 transactions_df: pd.DataFrame, 
                                                 save_path: Optional[str] = None) -> None:
        """
        Create visualizations for transaction analysis.
        
        Args:
            transactions_df: DataFrame containing transaction data
            save_path: Path to save the visualization (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Transaction Analysis Dashboard', fontsize=16)
        
        # Transaction Amount Distribution
        axes[0,0].hist(transactions_df['transaction_amount'], bins=50, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Transaction Amount Distribution')
        axes[0,0].set_xlabel('Amount ($)')
        axes[0,0].set_ylabel('Frequency')
        
        # Success vs Failure by Amount
        success_amounts = transactions_df[transactions_df['transaction_failure'] == 0]['transaction_amount']
        failure_amounts = transactions_df[transactions_df['transaction_failure'] == 1]['transaction_amount']
        axes[0,1].hist([success_amounts, failure_amounts], bins=30, label=['Success', 'Failure'], alpha=0.7)
        axes[0,1].set_title('Transaction Amount by Outcome')
        axes[0,1].set_xlabel('Amount ($)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Failure Rate by Transaction Type
        failure_by_type = transactions_df.groupby('transaction_type')['transaction_failure'].mean()
        axes[0,2].bar(failure_by_type.index, failure_by_type.values, color=['green' if x < 0.5 else 'red' for x in failure_by_type.values])
        axes[0,2].set_title('Failure Rate by Transaction Type')
        axes[0,2].set_xlabel('Transaction Type')
        axes[0,2].set_ylabel('Failure Rate')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Failure Rate by Time of Day
        failure_by_time = transactions_df.groupby('time_of_day')['transaction_failure'].mean()
        axes[1,0].plot(failure_by_time.index, failure_by_time.values, marker='o')
        axes[1,0].set_title('Failure Rate by Time of Day')
        axes[1,0].set_xlabel('Hour of Day')
        axes[1,0].set_ylabel('Failure Rate')
        
        # Failure Rate by Day of Week
        failure_by_day = transactions_df.groupby('day_of_week')['transaction_failure'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1,1].bar(range(7), failure_by_day.values, tick_label=days, 
                      color=['green' if x < 0.5 else 'red' for x in failure_by_day.values])
        axes[1,1].set_title('Failure Rate by Day of Week')
        axes[1,1].set_xlabel('Day of Week')
        axes[1,1].set_ylabel('Failure Rate')
        
        # Account Balance vs Transaction Amount
        scatter = axes[1,2].scatter(transactions_df['account_balance'], transactions_df['transaction_amount'], 
                                   c=transactions_df['transaction_failure'], cmap='viridis', alpha=0.6)
        axes[1,2].set_title('Account Balance vs Transaction Amount')
        axes[1,2].set_xlabel('Account Balance ($)')
        axes[1,2].set_ylabel('Transaction Amount ($)')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Transaction analysis visualizations saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, 
                                   transactions_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            transactions_df: DataFrame containing transaction data
            save_path: Path to save the dashboard as HTML (optional)
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transaction Amount Distribution', 
                           'Failure Rate by Transaction Type',
                           'Failure Rate by Time of Day',
                           'Success vs Failure Amount'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # Add transaction amount distribution
        fig.add_trace(
            go.Histogram(x=transactions_df['transaction_amount'], name='Amount Distribution'),
            row=1, col=1
        )
        
        # Add failure rate by transaction type
        failure_by_type = transactions_df.groupby('transaction_type')['transaction_failure'].mean()
        fig.add_trace(
            go.Bar(x=failure_by_type.index, y=failure_by_type.values, name='Failure by Type'),
            row=1, col=2
        )
        
        # Add failure rate by time of day
        failure_by_time = transactions_df.groupby('time_of_day')['transaction_failure'].mean()
        fig.add_trace(
            go.Scatter(x=failure_by_time.index, y=failure_by_time.values, 
                      mode='lines+markers', name='Failure by Time'),
            row=2, col=1
        )
        
        # Add box plot of amounts by success/failure
        success_amounts = transactions_df[transactions_df['transaction_failure'] == 0]['transaction_amount']
        failure_amounts = transactions_df[transactions_df['transaction_failure'] == 1]['transaction_amount']
        fig.add_trace(
            go.Box(y=success_amounts, name='Success', boxmean=True),
            row=2, col=2
        )
        fig.add_trace(
            go.Box(y=failure_amounts, name='Failure', boxmean=True),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Interactive Transaction Analysis Dashboard")
        
        # Show the figure
        fig.show()
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
    
    def save_report(self, report: Dict, filename: str) -> str:
        """
        Save a report to a JSON file.
        
        Args:
            report: Report dictionary to save
            filename: Name of the file to save to
            
        Returns:
            str: Path of the saved file
        """
        filepath = self.reports_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {filepath}")
        return str(filepath)
    
    def generate_monthly_report(self, 
                              transactions_df: pd.DataFrame, 
                              month: int, 
                              year: int) -> Dict[str, Any]:
        """
        Generate a monthly transaction report.
        
        Args:
            transactions_df: DataFrame containing transaction data
            month: Month for the report
            year: Year for the report
            
        Returns:
            dict: Monthly report
        """
        # Filter transactions for the specified month/year
        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
        monthly_transactions = transactions_df[
            (transactions_df['timestamp'].dt.month == month) & 
            (transactions_df['timestamp'].dt.year == year)
        ]
        
        if len(monthly_transactions) == 0:
            logger.warning(f"No transactions found for {year}-{month:02d}")
            return {
                'month': month,
                'year': year,
                'total_transactions': 0,
                'failed_transactions': 0,
                'success_rate': 0.0,
                'total_amount': 0.0,
                'summary': f"No transactions found for {year}-{month:02d}"
            }
        
        total_transactions = len(monthly_transactions)
        failed_transactions = monthly_transactions['transaction_failure'].sum()
        success_rate = 1 - (failed_transactions / total_transactions)
        total_amount = monthly_transactions['transaction_amount'].sum()
        
        report = {
            'month': month,
            'year': year,
            'total_transactions': total_transactions,
            'failed_transactions': int(failed_transactions),
            'success_rate': success_rate,
            'failure_rate': 1 - success_rate,
            'total_amount': total_amount,
            'avg_transaction_amount': monthly_transactions['transaction_amount'].mean(),
            'summary': f"""
            Monthly Report for {year}-{month:02d}
            ================================
            Total Transactions: {total_transactions}
            Failed Transactions: {int(failed_transactions)}
            Success Rate: {success_rate:.2%}
            Total Transaction Amount: ${total_amount:,.2f}
            Average Transaction Amount: ${monthly_transactions['transaction_amount'].mean():.2f}
            """
        }
        
        logger.info(f"Monthly report generated for {year}-{month:02d}")
        return report

def main():
    """Main function to demonstrate the analytics and reporting capabilities."""
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Import our modules
    from src.simulator.transaction_simulator import TransactionSimulator
    from datetime import datetime
    
    # Initialize the reporter
    reporter = AnalyticsReporter()
    
    # Generate sample transaction data
    print("Generating sample transaction data...")
    simulator = TransactionSimulator(failure_rate=0.15)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    transactions_df = simulator.generate_transactions(
        count=5000,
        start_date=start_date,
        end_date=end_date
    )
    
    # Generate transaction analysis report
    print("Generating transaction analysis report...")
    analysis_report = reporter.generate_transaction_analysis_report(transactions_df)
    report_path = reporter.save_report(analysis_report, "transaction_analysis_report.json")
    print(f"Transaction analysis report saved to {report_path}")
    
    # Generate mock prediction data for performance report
    print("Generating mock prediction data...")
    np.random.seed(42)  # For reproducible results
    y_true = transactions_df['transaction_failure'].tolist()
    # Simulate predictions with some accuracy
    y_pred = []
    for true_val in y_true:
        # 85% accuracy in predictions
        if np.random.random() < 0.85:
            y_pred.append(true_val)  # Correct prediction
        else:
            y_pred.append(1 - true_val)  # Incorrect prediction
    
    # Generate performance report
    print("Generating performance report...")
    performance_report = reporter.generate_performance_report(y_true, y_pred, model_name="Random Forest")
    perf_report_path = reporter.save_report(performance_report, "performance_report.json")
    print(f"Performance report saved to {perf_report_path}")
    
    # Create visualizations
    print("Creating visualizations...")
    reporter.create_prediction_visualizations(y_true, y_pred, save_path="reports/prediction_visualizations.png")
    reporter.create_transaction_analysis_visualizations(transactions_df, save_path="reports/transaction_analysis_visualizations.png")
    
    # Create interactive dashboard
    print("Creating interactive dashboard...")
    reporter.create_interactive_dashboard(transactions_df, save_path="reports/interactive_dashboard.html")
    
    # Generate monthly reports
    print("Generating monthly reports...")
    for month in range(1, 13):
        monthly_report = reporter.generate_monthly_report(transactions_df, month, 2023)
        monthly_report_path = reporter.save_report(monthly_report, f"monthly_report_{2023}_{month:02d}.json")
        print(f"Monthly report for 2023-{month:02d} saved to {monthly_report_path}")
    
    print("All reports and visualizations generated successfully!")

if __name__ == "__main__":
    main()