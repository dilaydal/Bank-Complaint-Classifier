import pandas as pd
import numpy as np
import nltk
from utils import process_text, build_freqs

nltk.download('stopwords')
nltk.download('punkt') #A tokenizer used to split sentences into words.


comp_file = pd.read_csv("filtered_complaints.csv")
texts = comp_file["text"].values
labels = comp_file["label"].values.reshape(-1, 1)

split_index = int(0.8 * len(texts))
training_texts = texts[:split_index]
testing_texts = texts[split_index:]

training_labels = labels[:split_index]
testing_labels = labels[split_index:]

freqs = build_freqs(training_texts, training_labels)

def extract_features(text, freqs): #text string to feature vector [bias, positive_word_count, negative_word_count]
    word_list = process_text(text)
    features = np.zeros(3)
    features[0] = 1

    for word in word_list:
        if (word, 1.0) in freqs:
            features[1] += freqs[(word, 1.0)]
        if (word, 0.0) in freqs:
            features[2] += freqs[(word, 0.0)]
    return features.reshape(1, -1)

train_features = []

for text in training_texts:
    features = extract_features(text, freqs)
    train_features.append(features) 

X_train = np.vstack(train_features) #list of 1x3 np arrays to (n, 3) matrix. n=complsints

test_features = [] #for testing

for text in testing_texts:
    features = extract_features(text, freqs)
    test_features.append(features)

X_test = np.vstack(test_features)


def sigmoid(z): #to model probabilities. any input -> (0, 1)
    return 1 / (1 + np.exp(-z))

def compute_cost(x, y, theta): #returns a single float value representing how wrong the model is right now.
    m = x.shape[0]
    h = sigmoid(np.dot(x, theta))
    return float(-1/m * (np.dot(y.T, np.log(h + 1e-15)) + np.dot((1 - y).T, np.log(1 - h + 1e-15))))

def gradient_descent(x, y, theta, alpha, num_iters): #trains the model by updating weights theta
    m = x.shape[0]
    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        gradient = np.dot(x.T, h - y) / m
        theta -= alpha * gradient
        if i % 100 == 0 or i == num_iters - 1:
            print(f"Iteration {i}: Cost = {compute_cost(x, y, theta):.5f}")
    return compute_cost(x, y, theta), theta

# Train
theta = np.zeros((3, 1))
final_cost, theta = gradient_descent(X_train, training_labels, theta, alpha=1e-7, num_iters=1500)

# === Evaluation ===
def predict(text, freqs, theta):
    x = extract_features(text, freqs)
    return sigmoid(np.dot(x, theta))[0, 0]

def test_logistic_regression(test_x, test_y, freqs, theta):
    correct = 0
    for text, label in zip(test_x, test_y):
        prob = predict(text, freqs, theta)
        pred = 1 if prob > 0.5 else 0
        if pred == label:
            correct += 1
    return correct / len(test_y)

acc = test_logistic_regression(testing_texts, testing_labels, freqs, theta)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print(freqs)
