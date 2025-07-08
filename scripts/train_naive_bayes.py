from wordcloud import WordCloud
import pandas as pd
import numpy as np
import nltk
from utils import process_text, build_freqs

nltk.download('stopwords')
nltk.download('punkt') 


comp_file = pd.read_csv("data/filtered_complaints.csv")
comp_file["word_count"] = comp_file["text"].apply(lambda x: len(process_text(x)))
comp_file.to_csv("data/filtered_complaints_with_word_count.csv", index=False)

texts = comp_file["text"].values
labels = comp_file["label"].values.reshape(-1, 1)

split_index = int(0.8 * len(texts))
training_texts = texts[:split_index]
testing_texts = texts[split_index:]

training_labels = labels[:split_index]
testing_labels = labels[split_index:]

freqs = build_freqs(training_texts, training_labels)


def train_naive_bayes(freqs, training_texts, training_labels):
    loglikelihood = {}
    logprior = 0

    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    
    D = len(training_labels) 
    D_pos = 0
    D_neg = 0

    for label in training_labels:
        if label == 1:
            D_pos += 1
        else:
            D_neg += 1

    #prior = D_pos / D_neg
    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        freq_pos = freqs.get((word, 1.0), 0)
        freq_neg = freqs.get((word, 0.0), 0)

        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = np.log(p_w_pos) -  np.log(p_w_neg) 


    return loglikelihood, logprior


def naive_bayes_predict(text, loglikelihood, logprior):
    word_list = process_text(text)
    p = 0
    p += logprior
    for word in word_list:
        if word in loglikelihood:
            p += loglikelihood[word]

    return p

def test_naive_bayes(testing_texts, testing_labels, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    accuracy = 0
    y_pred = []

    for text in testing_texts:
        if naive_bayes_predict(text, loglikelihood, logprior) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0

        y_pred.append(y_hat_i)

    total_dif = 0
    for y in range(len(y_pred)):
        total_dif += abs(y_pred[y] - testing_labels[y])
    error = total_dif / len(y_pred)
    accuracy = 1 - error
    return accuracy



def save_naive_bayes_errors(file_path, freqs, train_texts, train_labels):
    logprior, loglikelihood = train_naive_bayes(freqs, train_texts, train_labels)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Truth\tPredicted\tTweet\n")
        for x, y in zip(train_texts, train_labels):
            y_hat = naive_bayes_predict(x, logprior, loglikelihood)
            predicted = int(np.sign(y_hat) > 0)
            if y != predicted:
                processed = ' '.join(process_text(x))
                line = f"{int(y)}\t{predicted}\t{processed}\n"
                f.write(line)

    print(f"Naive Bayes misclassified some texts, saved to: {file_path}")



def save_naive_bayes_test_results(file_path, freqs, train_texts, train_labels, test_texts, test_labels):
    logprior, loglikelihood = train_naive_bayes(freqs, train_texts, train_labels)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Truth\tPredicted\tTweet\n")
        for x, y in zip(test_texts, test_labels):
            y_hat = naive_bayes_predict(x, logprior, loglikelihood)
            predicted = int(np.sign(y_hat) > 0)
            processed = ' '.join(process_text(x))
            line = f"{int(y)}\t{predicted}\t{processed}\n"
            f.write(line)

    print(f"Naive Bayes test results saved to: {file_path}")




save_naive_bayes_errors("data/naive_bayes_errors.txt", freqs, training_texts, training_labels)
print("Len training text: ", len(training_texts))
print("Len testing text: ", len(testing_texts))
print("Len complaints ", len(comp_file))
print("Len training labels: ", len(training_labels))
print("Len testing labels: ", len(testing_labels))
save_naive_bayes_test_results(
    file_path="data/naive_bayes_test_results.txt",
    freqs=freqs,
    train_texts=training_texts,
    train_labels=training_labels,
    test_texts=testing_texts,
    test_labels=testing_labels
)