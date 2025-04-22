import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load test data
x_test = np.load('ckd_app/data/x_test_data.npy')

# Create and fit scaler
scaler = StandardScaler()
scaler.fit(x_test)

# Save scaler
with open('ckd_app/models/scaler.sav', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler created and saved successfully!") 