import pandas as pd
import numpy as np
import nltk
from utils import process_text, build_freqs
nltk.download('stopwords')
nltk.download('punkt')
complaints = pd.read_csv("Bank_Complaint_Dataset.csv")

all_urgents = complaints[complaints['Urgent'] == 1]
all_nurgents = complaints[complaints['Urgent'] == 0]

train_complaints = complaints[:60]
urgent_train_complaints = train_complaints[train_complaints['Urgent'] == 1]
nurgent_train_complaints = train_complaints[train_complaints['Urgent'] == 0]    

test_complaints = complaints[60:]
urgent_test_complaints = test_complaints[test_complaints['Urgent'] == 1]
nurgent_test_complaints = test_complaints[test_complaints['Urgent'] == 0]

#Extract text and labels from train/test sets:
# Features (text)
train_texts = np.array(train_complaints["Complaint Text"])
test_texts = np.array(test_complaints["Complaint Text"])

# Labels (0 or 1 from 'Urgent' column)
train_labels = np.array(train_complaints["Urgent"])
test_labels = np.array(test_complaints["Urgent"])

freqs = build_freqs(train_texts, train_labels)
print(freqs)
print("train_texts:", train_texts)

def sigmoid(z): 
    h = 1 / (1 + np.exp(-z))
    return h

def compute_cost(x, y, theta):
    m = x.shape[0]
    z = np.dot(x, theta)
    h = sigmoid(z)
    J = -1/m * (np.dot(y.T, np.log(h + 1e-15)) + np.dot((1 - y).T, np.log(1 - h + 1e-15)))
    return float(J)

def gradient_descent(x, y, theta, alpha, num_iters):
    """
    x: feature matrix (m x n)
    y: label vector (m x 1)
    theta: weight vector (n x 1)
    alpha: learning rate
    num_iters: number of iterations
    Returns: final cost, final theta
    """
    m = x.shape[0]

    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        gradient = (1/m) * np.dot(x.T, (h - y))
        theta -= alpha * gradient
        
        if i % 100 == 0 or i == num_iters - 1:
            cost = compute_cost(x, y, theta)
            print(f"Iteration {i}: Cost = {cost:.5f}")
    
    final_cost = compute_cost(x, y, theta)
    return final_cost, theta

