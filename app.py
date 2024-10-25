import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to generate synthetic data
def create_synthetic_data(num_samples=1000):
    time = np.arange(num_samples)
    data = np.sin(0.1 * time) + 0.1 * np.random.randn(num_samples)  # Example sine wave data with noise
    return data

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# Streamlit app
st.title('LSTM Synthetic Tariff Prediction')

# Generate and prepare the data
data = create_synthetic_data()
data = data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences
SEQ_LENGTH = 10
X, y = create_sequences(data_scaled, SEQ_LENGTH)
X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, time steps, features)

# Load the pre-trained LSTM model
model = load_model('best_model.keras')

# Make predictions
predictions = model.predict(X)
predicted_tariffs = scaler.inverse_transform(predictions)
actual_tariffs = scaler.inverse_transform(y.reshape(-1, 1))

# Plotting results
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(actual_tariffs, label='Actual Tariffs', color='blue')
ax.plot(predicted_tariffs, label='Predicted Tariffs', color='red')
ax.set_title('Comparison of Actual and Predicted Tariffs')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Tariff Values')
ax.legend()

# Show the plot in Streamlit
st.pyplot(fig)

# Show predictions
st.write('Predicted Tariffs:')
st.dataframe(pd.DataFrame(predicted_tariffs, columns=['Predicted Tariff']))

