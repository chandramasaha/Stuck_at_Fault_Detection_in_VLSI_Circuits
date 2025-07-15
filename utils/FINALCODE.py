import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
import seaborn as sns

# Load the dataset
data_path = '/content/modified_c880.csv'
df = pd.read_csv(data_path)
df = df.fillna(0)

# Convert binary strings to numerical values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: [int(bit) for bit in x.strip() if bit in '01'])
        df[col] = df[col].apply(lambda x: sum(b * (2**i) for i, b in enumerate(reversed(x))))

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Split into training and test sets
X_train, X_test = train_test_split(data_scaled, test_size=0.25, random_state=42)

# Define Sparse Autoencoder
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(128, activation='relu', activity_regularizer=l1(1e-5))(input_layer)
encoded = Dropout(0.3)(encoded)
encoded = Dense(64, activation='relu', activity_regularizer=l1(1e-5))(encoded)
encoded = Dropout(0.3)(encoded)
bottleneck = Dense(114, activation='relu', activity_regularizer=l1(1e-5))(encoded)
decoded = Dropout(0.3)(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dropout(0.3)(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dropout(0.3)(decoded)
decoded = Dense(X_train.shape[1], activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mae')
autoencoder.summary()

# Train the model
history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=64, shuffle=True, validation_data=(X_test, X_test))

# Reconstruct full data
reconstructions = autoencoder.predict(data_scaled)
errors_all = np.mean(np.abs(data_scaled - reconstructions), axis=1)

# Train reconstruction loss
reconstructions_train = autoencoder.predict(X_train)
train_loss = np.mean(np.abs(X_train - reconstructions_train), axis=1)

# Define Ground Truth (synthetic: top 10% are faults)
ground_truth = np.zeros(len(errors_all))
gt_threshold = np.percentile(errors_all, 90)
ground_truth[errors_all > gt_threshold] = 1

# Method A: 95th Percentile of full evaluation data
percentile_threshold = np.percentile(errors_all, 95)
preds_A = (errors_all > percentile_threshold).astype(int)

# Method B1: Mean + 1 * Std of train loss
stat_threshold_1std = np.mean(train_loss) + 1 * np.std(train_loss)
preds_B1 = (errors_all > stat_threshold_1std).astype(int)

# Method B2: Mean + 2 * Std of train loss
stat_threshold_2std = np.mean(train_loss) + 2 * np.std(train_loss)
preds_B2 = (errors_all > stat_threshold_2std).astype(int)

# Method B3: 95th percentile of training loss
train_percentile_threshold = np.percentile(train_loss, 95)
preds_B3 = (errors_all > train_percentile_threshold).astype(int)

# Evaluation Function
def evaluate(preds, method_name, threshold):
    tp = np.sum((preds == 1) & (ground_truth == 1))
    tn = np.sum((preds == 0) & (ground_truth == 0))
    fp = np.sum((preds == 1) & (ground_truth == 0))
    fn = np.sum((preds == 0) & (ground_truth == 1))
    precision = precision_score(ground_truth, preds)
    recall = recall_score(ground_truth, preds)
    f1 = f1_score(ground_truth, preds)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n====== Evaluation: {method_name} ======")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")

# Evaluate All Methods
evaluate(preds_A, "Percentile ", percentile_threshold)
evaluate(preds_B1, "Mean + 1*Std ", stat_threshold_1std)
evaluate(preds_B2, "Mean + 2*Std ", stat_threshold_2std)


# Visualize distributions and thresholds
plt.figure(figsize=(10, 6))
sns.kdeplot(train_loss, label="Training Loss ", fill=True)
sns.kdeplot(errors_all, label="Validation Loss", fill=True)
plt.axvline(percentile_threshold, color='green', linestyle='--', label='Method A Threshold')
plt.axvline(stat_threshold_1std, color='red', linestyle='--', label='Method B1 (Mean + 1*Std)')
plt.axvline(stat_threshold_2std, color='purple', linestyle='--', label='Method B2 (Mean + 2*Std)')

plt.title("Reconstruction Error Distributions with Thresholds - c880 ")
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
