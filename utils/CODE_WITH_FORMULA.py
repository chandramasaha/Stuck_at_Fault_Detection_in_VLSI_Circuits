#FINAL FINAL CODE

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Preprocessing Functions
def preprocess_test_patterns(df):
    # Convert binary strings to lists of integers
    df['Test Pattern'] = df['Test Pattern'].apply(lambda x: [int(bit) for bit in x])
    return df

def convert_fault_free_response(df):
    # Convert binary strings to lists of integers
    df['Fault-Free Response'] = df['Fault-Free Response'].apply(lambda x: [int(bit) for bit in x])
    return df

# Load and Preprocess Data
df = pd.read_csv('modified_c432.csv', dtype={'Test Pattern': str, 'Fault-Free Response': str})
df = preprocess_test_patterns(df)
df = convert_fault_free_response(df)

# Extract test patterns (after preprocessing):
test_patterns = df['Test Pattern'].values
X_data = np.array([np.array(pattern) for pattern in test_patterns], dtype=np.float32)

# Check for consistent pattern lengths
pattern_lengths = [len(pattern) for pattern in test_patterns]
if len(set(pattern_lengths)) != 1:
    raise ValueError("All test patterns must have the same length.")

# Split Data into Training and Testing Sets
X_train, X_test = train_test_split(X_data, test_size=0.25, random_state=42)

# Define and Train Autoencoder with Enhanced Architecture
encoding_size = 524 # Set encoding size based on dataset complexity
input_layer = Input(shape=(X_train.shape[1],))

# Adding more hidden layers for better performance
encoding_layer = Dense(256, activation='relu')(input_layer)  # Increased units
encoding_layer = Dropout(0.3)(encoding_layer)  # Increased dropout
encoding_layer = Dense(128, activation='relu')(encoding_layer)
encoding_layer = Dropout(0.3)(encoding_layer)
encoding_layer = Dense(encoding_size)(encoding_layer)

decoding_layer = Dense(128, activation='relu')(encoding_layer)
decoding_layer = Dense(256, activation='relu')(decoding_layer)
decoding_layer = Dense(X_train.shape[1], activation='sigmoid')(decoding_layer)

autoencoder = Model(input_layer, decoding_layer)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='binary_crossentropy')
autoencoder.summary()

# Train Autoencoder with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = autoencoder.fit(X_train, X_train, epochs=25, batch_size=32,
                          validation_data=(X_test, X_test),
                          callbacks=[early_stopping])

# Evaluate reconstruction errors for the entire dataset
reconstructed_full = autoencoder.predict(X_data)
reconstruction_errors_full = np.mean(np.abs(X_data - reconstructed_full), axis=1)

mse = np.mean(np.square(X_data - reconstructed_full))
autoencoder_accuracy = 1 - mse

print(f"Autoencoder Accuracy (1 - MSE): {autoencoder_accuracy:.4f}")

# Define Reconstruction Error Threshold for Fault Detection
# Use mean + 2 * standard deviation to set a robust threshold
threshold = np.mean(reconstruction_errors_full) + 2 * np.std(reconstruction_errors_full)
print(f"Reconstruction Error Threshold: {threshold:.4f}")

# Classify Test Patterns as Faulty or Non-Faulty Based on Threshold
fault_detected = (reconstruction_errors_full > threshold).astype(int)

# Display Reconstruction Errors
df_results = pd.DataFrame({
    "Reconstruction Error": reconstruction_errors_full
})
print(df_results)

# Save results to CSV for reference
df_results.to_csv("reconstruction_errors.csv", index=False)
# Plot Reconstruction Errors and Threshold
plt.figure(figsize=(10, 6))
plt.scatter(range(len(reconstruction_errors_full)), reconstruction_errors_full, c=fault_detected, cmap='coolwarm', label='Fault Status')
plt.axhline(y=threshold, color='r', linestyle='--', label="Fault Detection Threshold")
plt.title('Reconstruction Errors and Fault Detection')
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()
# Filter reconstruction errors that exceed the threshold
errors_above_threshold = df_results[df_results["Reconstruction Error"] > threshold]

# Create the bar graph
plt.figure(figsize=(12, 6))
plt.scatter(errors_above_threshold.index, errors_above_threshold["Reconstruction Error"], color='red', label="Errors Above Threshold")
plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
plt.xlabel("Sample Index")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Errors Above Threshold")
plt.legend()
plt.show()