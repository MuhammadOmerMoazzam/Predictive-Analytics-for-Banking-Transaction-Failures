# AI-Based Transaction Failure Prediction and Analysis System

This project implements an AI-based system that predicts banking transaction failures and analyzes their causes. The system is designed to assist banks and financial institutions in reducing transaction losses, improving transaction success rates, and enhancing customer experience. The solution works as an intelligent middleware that analyzes transaction data, predicts failure probability, and generates analytical reports.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Integration](#integration)
- [Reporting](#reporting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This system implements a complete pipeline for predicting banking transaction failures using machine learning:

1. **Data Processing**: Load and preprocess transaction data
2. **Model Training**: Train ML models to predict transaction failures
3. **Model Evaluation**: Evaluate and optimize model performance
4. **API Service**: Provide prediction as a service via REST API
5. **Integration**: Connect with core banking platforms like Apache Fineract
6. **Analytics**: Generate reports and visualizations for transaction analysis

## Features

- **Multiple ML Algorithms**: Supports Logistic Regression, Random Forest, and Naive Bayes
- **Transaction Simulation**: Generate realistic synthetic transaction data
- **REST API**: FastAPI-based service for transaction failure predictions
- **Model Evaluation**: Comprehensive evaluation with accuracy, precision, recall, F1-score
- **Hyperparameter Optimization**: Automated optimization of model parameters
- **Apache Fineract Integration**: Connect with open-source core banking platform
- **Analytics Dashboard**: Visualizations and reports for transaction analysis
- **Real-time Predictions**: Fast prediction API for live transaction analysis

## Architecture

The system follows a modular architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Data Processing │───▶│  ML Model       │
│ (Public/Sim)    │    │  & Preprocessing │    │  Training       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌───────▼─────────┐
│ Transaction     │───▶│  Fineract        │───▶│  Model          │
│ Simulator       │    │  Integration     │    │  Evaluation     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌───────▼─────────┐
│ API Service     │    │  Analytics &     │    │  Prediction     │
│ (FastAPI)       │◀───│  Reporting       │    │  Service        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd failure-transaction-algo
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### 1. Train a Model

```bash
python -m src.model.train_model --data_path data/your_transaction_data.csv
```

### 2. Start the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_amount": 100.50,
       "account_balance": 2500.00,
       "time_of_day": 14,
       "day_of_week": 2,
       "transaction_type": "debit",
       "location": "local",
       "merchant_category": "grocery",
       "location_risk_score": 0.2,
       "historical_failure_rate": 0.05
     }'
```

### 4. Generate Synthetic Transactions

```bash
python -m src.simulator.transaction_simulator --count 10000 --output-file data/synthetic_transactions.csv
```

### 5. Run Analytics

```bash
python -m src.analytics.reporting_module
```

### 6. Using Jupyter Notebooks

To run the Jupyter notebooks successfully, you need to run the setup cell first. The notebooks are located in the `notebooks/` directory, but the Python modules are in the project root directory, so special path configuration is needed:

1. Start Jupyter from the project root directory:
   ```bash
   # Navigate to the project root directory first
   cd D:\AI_Semester_Project\failure-transaction-algo
   jupyter notebook
   ```

2. Or, if already in Jupyter, run this setup cell at the beginning of any notebook:
   ```python
   import sys, os

   # Get the parent directory (project root) since notebooks are in the 'notebooks' subdirectory
   project_root = os.path.dirname(os.getcwd())  # Go up one level from notebooks/

   # Add the project root to Python path if not already there
   if project_root not in sys.path:
       sys.path.insert(0, project_root)

   print(f"Project root added to path: {project_root}")
   ```

3. Then you can import modules:
   ```python
   from src.simulator.transaction_simulator import TransactionSimulator
   from src.model.data_processor import preprocess_data
   # ... other imports
   ```

4. Alternatively, you can use the troubleshooting notebook we've provided:
   - Run `notebooks/troubleshooting_imports.ipynb` first to set up the paths
   - This notebook automatically detects and fixes import issues

### 7. Troubleshooting

If you encounter import errors, run the troubleshooting notebook:
```bash
jupyter notebook notebooks/troubleshooting_imports.ipynb
```

### 8. Project Structure

```
failure-transaction-algo/
├── src/                     # Source code
│   ├── model/              # ML model training and evaluation
│   ├── simulator/          # Transaction simulator
│   ├── api/                # FastAPI implementation
│   ├── integration/        # Fineract integration
│   └── analytics/          # Reporting and analytics
├── notebooks/              # Jupyter notebooks for demos
├── data/                   # Data files
├── models/                 # Trained models
├── reports/                # Generated reports
├── specs/                  # Project specifications
│   └── transaction-failure-prediction/
│       ├── spec.md         # Project specification
│       ├── plan.md         # Implementation plan
│       └── tasks.md        # Task breakdown
├── history/                # Speckit Plus history
│   ├── prompts/            # Prompt history records
│   └── adr/                # Architectural decision records
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .env.example           # Environment variables example
```

## Development

### Running Tests

```bash
pytest tests/
```

### Jupyter Notebooks

Several notebooks demonstrate different aspects of the system:

- `notebooks/model_training_demo.ipynb` - Model training process
- `notebooks/transaction_simulator_demo.ipynb` - Transaction simulation
- `notebooks/model_evaluation_demo.ipynb` - Model evaluation and optimization
- `notebooks/api_demo.ipynb` - API usage examples
- `notebooks/fineract_integration_demo.ipynb` - Fineract integration
- `notebooks/reporting_analytics_demo.ipynb` - Analytics and reporting

### Environment Variables

The system uses the following environment variables:

- `MODEL_PATH` - Path to the trained model file
- `SCALER_PATH` - Path to the feature scaler file
- `API_HOST` - Host for the API server (default: 0.0.0.0)
- `API_PORT` - Port for the API server (default: 8000)
- `FINERACT_URL` - URL for Apache Fineract instance
- `FINERACT_USERNAME` - Username for Fineract authentication
- `FINERACT_PASSWORD` - Password for Fineract authentication

## Testing

The system includes comprehensive testing:

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **API Tests**: Test API endpoints
4. **Model Tests**: Validate model performance

To run tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model_training.py

# Run tests with coverage
pytest --cov=src/
```

## API Documentation

The API provides the following endpoints:

### `/`
- **Method**: GET
- **Description**: API root endpoint
- **Response**: API status information

### `/health`
- **Method**: GET
- **Description**: Health check endpoint
- **Response**: Health status of the API and loaded models

### `/predict`
- **Method**: POST
- **Description**: Predict transaction failure
- **Request Body**: Transaction details
- **Response**: Failure probability and prediction

### `/predict_batch`
- **Method**: POST
- **Description**: Predict failures for multiple transactions
- **Request Body**: List of transaction details
- **Response**: List of predictions

### `/model_info`
- **Method**: GET
- **Description**: Get information about the loaded model
- **Response**: Model type and metadata

### `/docs`
- **Method**: GET
- **Description**: Interactive API documentation (Swagger UI)

### `/redoc`
- **Method**: GET
- **Description**: Alternative API documentation (ReDoc)

### `/reports/transaction-analysis`
- **Method**: GET
- **Description**: Get transaction analysis report
- **Response**: Summary of transaction patterns and failure rates

### `/reports/performance-metrics`
- **Method**: GET
- **Description**: Get model performance metrics report
- **Response**: Accuracy, precision, recall, F1-score, and other metrics

### `/reports/monthly-summary/{year}/{month}`
- **Method**: GET
- **Description**: Get monthly transaction summary report
- **Parameters**:
  - `year`: Year for the report (e.g., 2023)
  - `month`: Month for the report (1-12)
- **Response**: Monthly transaction summary with failure analysis

### `/reports/generate-full-report`
- **Method**: GET
- **Description**: Generate comprehensive analytics report
- **Response**: Complete analytics report with recommendations

### Frontend Integration
The Failure-Shield frontend connects to the backend API through proxy routes:
- **Analytics Dashboard**: Displays transaction analysis, performance metrics, and recommendations
- **API Key Management**: Secure API key generation and management
- **Real-time Reporting**: Live data from the prediction models

## Integration

### Apache Fineract Integration

The system can integrate with Apache Fineract to retrieve real transaction data:

1. Set up Apache Fineract server
2. Configure authentication credentials
3. Use the `FineractIntegration` class to retrieve transaction data
4. Transform Fineract data to model input format
5. Make predictions using the API

Example:
```python
from src.integration.fineract_integration import FineractIntegration

# Initialize integration
fineract = FineractIntegration(
    base_url="http://localhost:8080/fineract-provider/api/v1/",
    username="mifos",
    password="password",
    tenant_id="default"
)

# Authenticate
fineract.authenticate()

# Get client transactions
transactions = fineract.get_client_transactions(client_id=1)

# Transform and predict
for transaction in transactions:
    transformed = fineract.transform_fineract_transaction(transaction)
    # Make prediction using the API
```

## Reporting

The system generates comprehensive reports:

1. **Performance Reports**: Model accuracy, precision, recall, F1-score
2. **Transaction Analysis**: Failure rates by type, location, time, etc.
3. **Monthly Reports**: Trends over time
4. **Visualizations**: Charts and graphs for data analysis
5. **Recommendations**: AI-based suggestions for improving success rates

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- New features include appropriate tests
- Documentation is updated

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Apache Fineract for the open-source core banking platform
- Scikit-learn for machine learning capabilities
- FastAPI for the web framework
- Pandas and NumPy for data processing
- Plotly and Matplotlib for visualization