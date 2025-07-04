# Bank Complaint Dispute Classifier

This project implements a **logistic regression classifier** to predict whether a bank complaint was **disputed** by the consumer or not.

---

## Dataset

Data comes from the [Consumer Financial Protection Bureau (CFPB)](https://www.kaggle.com/code/shtrausslearning/banking-consumer-complaint-analysis). Each entry includes:

- `text`: The complaint narrative  
- `label`:  
  - `1` → Disputed complaint  
  - `0` → Not disputed

Processed data files:
- `filtered_complaints.csv` — Cleaned and labeled complaints  
- `filtered_complaints_with_word_counts.csv` — Same, but with an extra column for processed word count

---

## Approach

- **Preprocessing**:  
  Lowercasing, tokenization, stemming, and stopword removal using NLTK.

- **Features**:  
  Each complaint is represented as a vector:  
  `[bias_term, disputed_word_count, non_disputed_word_count]`

- **Model**:  
  Logistic regression, trained via gradient descent.

---

## File Structure

```
BankComplaints/
│
├── filtered_complaints.csv                     # Preprocessed labeled data
├── filtered_complaints_with_word_counts.csv    # Same, with word counts
├── train_model.py                              # Trains the model
├── utils.py                                    # Text cleaning & frequency logic
├── requirements.txt                            # Python package dependencies
├── README.md                                   # This file
└── wordcloud.png                               # Word cloud of all complaint words
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas nltk matplotlib wordcloud
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Train a logistic regression model
- Print cost updates every 100 iterations
- Report test accuracy
- Save a word cloud image to `wordcloud.png`


This will prompt you for a complaint text and return the prediction:  
**Disputed (1)** or **Not disputed (0)**.

---

## Word Cloud

A word cloud of all complaint words is generated during training and saved to:

```
wordcloud.png
```

---

## Feature Summary

Each complaint is converted to a vector:
```
[1, count_of_disputed_words, count_of_non_disputed_words]
```

This simple representation feeds into a logistic regression model that performs well with clean, labeled data.
