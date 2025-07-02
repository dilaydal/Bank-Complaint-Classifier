Bank Complaint Dispute Classifier

This project builds a **logistic regression classifier** to predict whether a bank complaint was **disputed** by the consumer or not, using a manually engineered frequency-based feature set (no TF-IDF).

## Dataset

The dataset is derived from [Consumer Financial Protection Bureau (CFPB)](https://www.kaggle.com/code/shtrausslearning/banking-consumer-complaint-analysis) complaints, where each entry includes:

- `text`: The narrative of the consumer's complaint
- `label`: Whether the consumer **disputed** the complaint (`1`) or not (`0`)

Filtered data is saved as:  
filtered_complaints.csv

## Approach

- Preprocessing includes tokenization, lowercasing, stemming, and stopword removal using NLTK.
- Features are built by counting how often each word appears in **disputed** and **not disputed** complaints.
- A **logistic regression** model is trained using these features to predict dispute likelihood.

## File Structure

BankComplaints/
│
├── filtered_complaints.csv # Preprocessed and filtered labeled dataset
├── train_model.py # Trains the logistic regression classifier
├── predict_custom.py # Lets you try your own complaint predictions
├── utils.py # Text processing and feature extraction helpers
├── README.md 

## How to Run

### Install dependencies

pip install numpy pandas nltk

Train the model:
python train_model.py

This trains the model and shows learning cost over time and test accuracy.

Predict custom complaints
python predict_custom.py