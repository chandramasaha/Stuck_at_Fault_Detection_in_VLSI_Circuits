#FINAL CODE FOR 880 with clusters
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Preprocessing Functions (same as before)

# Load and Preprocess Data
df = pd.read_csv('modified_c432.csv', dtype={'Test Pattern': str, 'Fault-Free Response': str})
df = preprocess_test_patterns(df)
df = convert_fault_free_response(df)

# Extract test patterns (after preprocessing):
test_patterns = df['Test Pattern'].values
X_data = np.array([np.array(pattern) for pattern in test_patterns], dtype=np.float32)

# Split Data into Training and Testing Sets
X_train, X_test = train_test_split(X_data, test_size=0.25, random_state=42)

# Define and Train Autoencoder with Enhanced Architecture for c880
encoding_size = 542  # Set encoding size based on dataset complexity
input_layer = Input(shape=(X_train.shape[1],))

# Adding more hidden layers for better performance
encoding_layer = Dense(256, activation='relu')(input_layer)  # Increased units
encoding_layer = Dropout(0.3)(encoding_layer)  # Increased dropout
encoding_layer = Dense(128, activation='relu')(encoding_layer)
encoding_layer = Dense(encoding_size)(encoding_layer)

decoding_layer = Dense(128, activation='relu')(encoding_layer)
decoding_layer = Dense(256, activation='relu')(decoding_layer)
decoding_layer = Dense(X_train.shape[1], activation='sigmoid')(decoding_layer)

autoencoder = Model(input_layer, decoding_layer)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='binary_crossentropy')
autoencoder.summary()


# Train Autoencoder
history = autoencoder.fit(X_train, X_train, epochs=25, batch_size=32,
                          validation_data=(X_test, X_test))

# Evaluate reconstruction errors for the entire dataset
reconstructed_full = autoencoder.predict(X_data)
reconstruction_errors_full = np.mean(np.abs(X_data - reconstructed_full), axis=1)

mse = np.mean(np.square(X_data - reconstructed_full))
autoencoder_accuracy = 1 - mse

print(f"Autoencoder Accuracy (1 - MSE): {autoencoder_accuracy:.4f}")


# Scale Reconstruction Errors for Clustering
scaler = StandardScaler()
scaled_errors = scaler.fit_transform(reconstruction_errors_full.reshape(-1, 1))

# Determine Optimal Number of Clusters Using Silhouette Score (Optional)
silhouette_scores = []
for k in range(2, 6):  # Test k values from 2 to 5
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans_temp.fit(scaled_errors)
    silhouette_scores.append(silhouette_score(scaled_errors, kmeans_temp.labels_))

optimal_k = np.argmax(silhouette_scores) + 2  # Add 2 because range starts at k=2

print(f"Optimal number of clusters determined by Silhouette Score: {optimal_k}")

# Apply K-means Clustering with Optimal Number of Clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(scaled_errors)

# Determine which cluster corresponds to faults (assumed higher error means faulty)
faulty_cluster_label = np.argmax(kmeans.cluster_centers_).astype(int)


# Save results to CSV for reference
df_results.to_csv("reconstruction_errors.csv", index=False)
# Plot K-means Clustering Results (Visualization of Clusters)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(scaled_errors)), scaled_errors, c=cluster_labels, cmap='viridis', label='Clusters')
plt.axhline(y=kmeans.cluster_centers_[faulty_cluster_label], color='r', linestyle='--', label="Faulty Cluster Center")
plt.title('K-means Clustering of Reconstruction Errors')
plt.xlabel('Sample Index')
plt.ylabel('Scaled Reconstruction Error')
plt.legend()
plt.show()

# Create DataFrame for results with cluster labels and fault detection status
df_results = pd.DataFrame({
    "Reconstruction Error": reconstruction_errors_full,
    "Cluster Label": cluster_labels,
    "Fault Detected": (cluster_labels == np.argmax(kmeans.cluster_centers_)).astype(int)
})
# Filter reconstruction errors that exceed the threshold
errors_above_threshold = df_results[df_results["Reconstruction Error"] > threshold]
# Display Reconstruction Errors
df_results = pd.DataFrame({
    "Reconstruction Error": reconstruction_errors_full
})
print(df_results)

# Save results to CSV for reference
df_results.to_csv("reconstruction_errors.csv", index=False)

