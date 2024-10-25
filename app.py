import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to create synthetic data
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
st.title('Tariff Prediction App')

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
try:
    model = load_model('best_model.keras')  # Load your model here
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# If model loading was successful, proceed with predictions
if 'model' in locals():
    # Make predictions
    predictions = model.predict(X)

    # Inverse transform and ensure non-negative values
    predicted_tariffs = scaler.inverse_transform(predictions)
    predicted_tariffs = np.abs(predicted_tariffs)  # Ensure predicted tariffs are non-negative

    # Inverse transform actual tariffs and ensure non-negative values
    actual_tariffs = scaler.inverse_transform(y.reshape(-1, 1))
    actual_tariffs = np.abs(actual_tariffs)  # Ensure actual tariffs are non-negative

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

    # Additional Plot: Highlight Low and High Tariff Regions
    threshold = np.mean(actual_tariffs)  # Set a threshold as the mean of actual tariffs
    low_tariff_indices = np.where(actual_tariffs < threshold)[0]
    high_tariff_indices = np.where(actual_tariffs >= threshold)[0]

    # Plotting low and high tariff regions
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(actual_tariffs, label='Actual Tariffs', color='gray')
    ax2.scatter(low_tariff_indices, actual_tariffs[low_tariff_indices], color='green', label='Low Tariff', marker='o', alpha=0.6)
    ax2.scatter(high_tariff_indices, actual_tariffs[high_tariff_indices], color='red', label='High Tariff', marker='x', alpha=0.6)  # Corrected variable name
    ax2.axhline(threshold, color='orange', linestyle='--', label='Mean Tariff Threshold')
    ax2.set_title('Low and High Tariff Regions')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Tariff Values')
    ax2.legend()

    # Show the second plot in Streamlit
    st.pyplot(fig2)
