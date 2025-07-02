import numpy as np
import pandas as pd
from utils import process_text, build_freqs
from train_model import extract_features, sigmoid

# Load the CSV and model
df = pd.read_csv("filtered_complaints.csv")
texts = df["text"].values
labels = df["label"].values.reshape(-1, 1)

# Load the frequency dictionary and weights
split_idx = int(0.8 * len(texts))
train_texts = texts[:split_idx]
train_labels = labels[:split_idx]

# Build freqs again
freqs = build_freqs(train_texts, train_labels)

# Load trained theta manually if saved, or re-train (simpler for now)
# Tip: In production, save theta to a file after training

# === Recreate training features and retrain to get theta ===
X_train = np.vstack([extract_features(t, freqs) for t in train_texts])
theta = np.zeros((3, 1))

def gradient_descent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        gradient = np.dot(x.T, h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(X_train, train_labels, theta, alpha=1e-7, num_iters=1500)

# === Predict on Custom Text ===
def predict(text, freqs, theta):
    x = extract_features(text, freqs)
    return sigmoid(np.dot(x, theta))[0, 0]

# === Try User Input ===
while True:
    text = input("\nEnter a complaint: ")
    prob = predict(text, freqs, theta)
    print(f"Predicted probability of being disputed: {prob:.4f}")
    print("Prediction:", "DISPUTED (1)" if prob > 0.5 else "NOT DISPUTED (0)")
