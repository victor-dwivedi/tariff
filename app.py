import numpy as np
import pandas as pd
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io

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

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=64)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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

# Split the data into training and validation sets
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Build and train the model
model = build_model((X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Make predictions
predictions = model.predict(X_val)
predicted_tariffs = scaler.inverse_transform(predictions)
actual_tariffs = scaler.inverse_transform(y_val.reshape(-1, 1))

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
