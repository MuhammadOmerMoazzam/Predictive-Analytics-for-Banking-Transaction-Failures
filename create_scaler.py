import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create a scaler and fit it with some sample data
scaler = StandardScaler()
# Using 9 features as per the transaction data
sample_data = np.random.rand(100, 9)
scaler.fit(sample_data)

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Create a dummy label encoders dict
label_encoders = {}
joblib.dump(label_encoders, 'models/label_encoders.pkl')