# Bank Complaint Dispute Classifier

This project implements **Logistic Regression** and **Naive Bayes** classifiers to predict whether a bank complaint was **disputed** by the consumer or not.

---

## Dataset

Data is sourced from the [Consumer Financial Protection Bureau (CFPB)](https://www.kaggle.com/code/shtrausslearning/banking-consumer-complaint-analysis). Each entry includes:

- `text`: The complaint narrative  
- `label`:  
  - `1` → Disputed complaint  
  - `0` → Not disputed

Processed files:
- `filtered_complaints.csv` — Cleaned and labeled complaints  
- `filtered_complaints_with_word_count.csv` — Same, with added word count  
- `logistic_results.txt`, `naive_bayes_test_results.txt` — Prediction logs for test set  
- `logistic_errors.txt`, `naive_bayes_errors.txt` — Misclassified examples

---

## Models

### 1. Logistic Regression  
- Feature vector:  
  `[bias_term, disputed_word_count, non_disputed_word_count]`  
- Trained using gradient descent  
- Sigmoid activation and cost minimization  
- Predicts via thresholding (prob > 0.5 → disputed)

### 2. Naive Bayes  
- Trained on word frequencies in disputed vs. non-disputed complaints  
- Predicts based on log-likelihood of word occurrence in each class  
- Fast and interpretable baseline model

---

## Project Structure
```
BankComplaints/
├── data/
│   ├── filtered_complaints.csv
│   ├── filtered_complaints_with_word_count.csv
│   ├── logistic_results.txt
│   ├── naive_bayes_results.txt
│   ├── logistic_errors.txt
│   └── naive_bayes_errors.txt
│
├── scripts/
│   ├── train_logistic.py
│   ├── train_naive_bayes.py
│   └── utils.py
│
├── visuals/
│   └── wordcloud.png
│
├── requirements.txt
└── README.md
```
## Usage

### Environment Setup

Install dependencies:

pip install -r requirements.txt

### Training

Train Logistic Regression:

python scripts/train_logistic.py

Train Naive Bayes:

python scripts/train_naive_bayes.py

### Output

Training results and classification errors are saved in:

- Logistic Regression: data/logistic_results.txt and data/logistic_errors.txt

- Naive Bayes: data/naive_bayes_results.txt and data/naive_bayes_errors.txt

### Visualization

Word cloud visualization of frequent terms:

See visuals/wordcloud.png

### Dependencies

Key Python libraries used:

NLTK
NumPy
pandas
matplotlib
wordcloud