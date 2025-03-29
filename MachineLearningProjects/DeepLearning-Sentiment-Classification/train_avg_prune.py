#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from collections import Counter

# Load word embeddings
def load_embeddings(embedding_file):
    return KeyedVectors.load(embedding_file)

# Read data
def read_from(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        yield (1 if label == "+" else -1, words.split())

# Prune words that appear only once
def prune_one_count_words(trainfile):
    word_counts = Counter()
    for _, words in read_from(trainfile):
        word_counts.update(words)
    # Keep only words that appear more than once
    valid_words = {word for word, count in word_counts.items() if count > 1}
    return valid_words

# Generate dense vector (average word embeddings with a bias term)
def make_dense_vector_with_bias(words, wv, valid_words):
    # Filter out one-count words
    vectors = [wv[word] for word in words if word in valid_words and word in wv]
    if not vectors:
        avg_vector = np.zeros(wv.vector_size)
    else:
        avg_vector = np.mean(vectors, axis=0)
    return np.append(avg_vector, 1)  # Add bias as the last dimension

# Test averaged perceptron with dense embeddings and pruning
def test_dense_with_pruning(devfile, model, wv, valid_words):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1):
        sent_vector = make_dense_vector_with_bias(words, wv, valid_words)  # Dense vector with pruning
        prediction = np.dot(model, sent_vector)  # Compute dot product
        err += label * prediction <= 0  # Count errors
    return err / i  # i is |D| now

# Averaged Perceptron with pruning one-count words
def train_avg_perceptron_with_pruning(trainfile, devfile, wv, epochs=5):
    # Prune one-count words
    valid_words = prune_one_count_words(trainfile)

    # Initialize model and cumulative weights
    model = np.zeros(wv.vector_size + 1)  # Current weights, including bias
    avg_model = np.zeros(wv.vector_size + 1)  # Cumulative weights
    c = 0  # Counter to track total number of updates

    best_model = None
    best_err = 1.0

    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent_vector = make_dense_vector_with_bias(words, wv, valid_words)  # Dense vector with pruning
            prediction = np.dot(model, sent_vector)  # Compute dot product
            if label * prediction <= 0:  # Misclassified
                updates += 1
                model += label * sent_vector  # Update weights
            avg_model += model  # Accumulate weights
            c += 1
        # Test the averaged model on the development set
        averaged_model = avg_model / c  # Average the weights
        dev_err = test_dense_with_pruning(devfile, averaged_model, wv, valid_words)
        if dev_err < best_err:
            best_err = dev_err
            best_model = averaged_model.copy()  # Save the best model
        print(f"Epoch {it}, updates {updates / i * 100:.1f}%, dev error {dev_err * 100:.1f}%")
    print(f"Best dev error {best_err * 100:.1f}%")
    return best_model

# Main entry point
if __name__ == "__main__":
    embedding_file = "embs_train.kv"  # Replace with your embedding file path
    trainfile = sys.argv[1]
    devfile = sys.argv[2]

    # Load embeddings
    wv = load_embeddings(embedding_file)

    # Train and test averaged perceptron with pruning
    best_model = train_avg_perceptron_with_pruning(trainfile, devfile, wv, epochs=10)

    # Save predictions for Kaggle submission
    test_data = pd.read_csv('test.csv')
    predictions = []
    valid_words = prune_one_count_words(trainfile)  # Prune words in test predictions as well
    for i, row in test_data.iterrows():
        sent_vector = make_dense_vector_with_bias(row['sentence'].split(), wv, valid_words)
        predicted_label = '+' if np.dot(best_model, sent_vector) > 0 else '-'
        predictions.append(predicted_label)
    test_data['target'] = predictions
    test_data.to_csv('test.predicted.avg.prune.csv', index=False)
    print(f"Predictions saved to test.predicted.csv")
