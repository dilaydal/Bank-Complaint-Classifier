from collections import Counter
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from utils import process_text, build_freqs

nltk.download('stopwords')
nltk.download('punkt') #A tokenizer used to split sentences into words.


comp_file = pd.read_csv("filtered_complaints.csv")
comp_file["word_count"] = comp_file["text"].apply(lambda x: len(process_text(x)))
comp_file.to_csv("filtered_complaints_with_word_count.csv", index=False)

texts = comp_file["text"].values
labels = comp_file["label"].values.reshape(-1, 1)

split_index = int(0.8 * len(texts))
training_texts = texts[:split_index]
testing_texts = texts[split_index:]

training_labels = labels[:split_index]
testing_labels = labels[split_index:]

# train_labels_flat = training_labels.flatten().tolist()
# test_labels_flat = testing_labels.flatten().tolist()
# train_distribution = Counter(train_labels_flat)
# test_distribution = Counter(test_labels_flat)
# print("Training label distribution:", train_distribution)
# print("Testing label distribution:", test_distribution)
# total_distribution = Counter(train_labels_flat + test_labels_flat)
# print("Total label distribution (should match CSV):", total_distribution)

freqs = build_freqs(training_texts, training_labels)

def extract_features(text, freqs): #text string to feature vector [bias, positive_word_count, negative_word_count]
    word_list = process_text(text)
    #print("text:", text)    
    features = np.zeros(3)
    features[0] = 1
    for word in word_list:
        if (word, 1.0) in freqs:
            features[1] += freqs[(word, 1.0)]
        if (word, 0.0) in freqs:
            features[2] += freqs[(word, 0.0)]
    features[1] = np.log1p(features[1])
    features[2] = np.log1p(features[2])
    return features.reshape(1, -1) #(n_features,) to (1, n_features) shape

train_features = []

for text in training_texts:
    features = extract_features(text, freqs)
    train_features.append(features) 

X_train = np.vstack(train_features) #list of 1x3 np arrays to (n, 3) matrix. n=complsints

test_features = []

for text in testing_texts:
    features = extract_features(text, freqs)
    test_features.append(features)

X_test = np.vstack(test_features)


def sigmoid(z): # converts z to a probability between 0 and 1.. any input -> (0, 1)
    z = np.clip(z, -500, 500)
    p = 1 / (1 + np.exp(-z))
    #print("sigmoid output:", p)
    return p

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
final_cost, theta = gradient_descent(X_train, training_labels, theta, alpha=1e-3, num_iters=5000)

def predict(text, freqs, theta):
    x = extract_features(text, freqs) #text to feature vector, x here
    return sigmoid(np.dot(x, theta))[0, 0] # np.dot(x, theta) = z, sigmoid(z) = probability

def interactive_predict_loop():
    while True:
        text = input("\nEnter a complaint: ")
        prob = predict(text, freqs, theta)
        print(f"\nText: {text}")
        print(f"Extracted features: {extract_features(text, freqs)}")
        print(f"Model parameters (theta): {theta.flatten()}")
        print(f"Predicted probability of being disputed: {prob:.4f}")
        if prob < 0.5:
            print("Prediction:", "NOT DISPUTED (0)")
        else:
            print("Prediction:", "DISPUTED (1)")

def test_logistic_regression(test_texts, test_labels, freqs, theta):
    correct = 0
    total = len(test_labels)
    for text, label in zip(test_texts, test_labels):
        predicted_prob = predict(text, freqs, theta)
        if predicted_prob > 0.5:
            predicted_label = 1 
        else:
            predicted_label = 0 

        if predicted_label == label:
            correct += 1
    return correct / total

def generate_wordcloud(freqs, label=None, title="Word Cloud", output_path="wordcloud.png"):
    word_freq = {}

    for (word, lbl), count in freqs.items():
        if label is None or lbl == label:
            word_freq[word] = word_freq.get(word, 0) + count

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Word cloud saved as: {output_path}")
    plt.close()

acc = test_logistic_regression(testing_texts, testing_labels, freqs, theta)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
generate_wordcloud(freqs, title="All Complaints")
interactive_predict_loop()