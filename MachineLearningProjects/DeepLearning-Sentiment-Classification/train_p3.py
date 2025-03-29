#!/usr/bin/env python3

import sys
import time
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Hard-coded embedding file path
EMBEDDING_FILE = "embs_train.kv"

# Load word embeddings
def load_embeddings():
    return KeyedVectors.load(EMBEDDING_FILE)

# Negation handling
def handle_negation(sentence):
    negation_words = {"not", "n't", "never", "no"}
    words = sentence.split()
    new_words = []
    negation_active = False

    for word in words:
        if word in negation_words:
            negation_active = True
            new_words.append(word)
        elif negation_active:
            new_words.append(f"NOT_{word}")
            negation_active = False
        else:
            new_words.append(word)

    return " ".join(new_words)

# Preprocess data with negation handling
def preprocess_data(textfile):
    data = pd.read_csv(textfile)
    sentences, labels = [], []
    for _, row in data.iterrows():
        sentence = handle_negation(row['sentence'])
        sentences.append(sentence.split())  # Split into words
        labels.append(1 if row['target'] == "+" else 0)
    return sentences, labels

# Generate dense vectors from embeddings
def make_dense_vector_with_bias(words, wv):
    word_vectors = [wv[word] for word in words if word in wv]
    avg_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(wv.vector_size)
    return np.append(avg_vector, 1)  # Add bias

# Generate dense vectors for a list of sentences
def generate_dense_vectors(sentences, wv):
    return np.array([make_dense_vector_with_bias(sentence, wv) for sentence in sentences])

# Train and evaluate Logistic Regression model
def train_and_evaluate_model(X_train, y_train, X_dev, y_dev):
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    print("Training Logistic Regression...")
    train_start_time = time.time()
    model.fit(X_train, y_train)
    train_end_time = time.time()

    # Evaluate on the development set
    dev_predictions = model.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, dev_predictions)
    dev_error_rate = 1 - dev_accuracy

    print(f"Logistic Regression - Dev Error Rate: {dev_error_rate * 100:.2f}%")
    print(f"Training time: {train_end_time - train_start_time:.2f} seconds")
    return model, train_end_time - train_start_time

# Save predictions for submission
def save_predictions(test_file, model, wv, output_file="test.predicted.csv"):
    test_data = pd.read_csv(test_file)
    sentences = [row['sentence'].split() for _, row in test_data.iterrows()]
    X_test = generate_dense_vectors(sentences, wv)
    predictions = model.predict(X_test)
    test_data['target'] = ['+' if pred == 1 else '-' for pred in predictions]
    test_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Main entry point
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <trainfile> <devfile>")
        sys.exit(1)

    trainfile = sys.argv[1]
    devfile = sys.argv[2]
    testfile = "test.csv"

    # Start timing the whole process
    total_start_time = time.time()

    # Load embeddings
    print(f"Loading embeddings from {EMBEDDING_FILE}...")
    wv = load_embeddings()

    # Preprocess training and development data
    print("Processing data...")
    train_sentences, train_labels = preprocess_data(trainfile)
    dev_sentences, dev_labels = preprocess_data(devfile)

    # Generate dense vectors
    print("Generating dense vectors...")
    train_vectors = generate_dense_vectors(train_sentences, wv)
    dev_vectors = generate_dense_vectors(dev_sentences, wv)

    # Train and evaluate Logistic Regression
    logistic_model, training_time = train_and_evaluate_model(train_vectors, train_labels, dev_vectors, dev_labels)

    # Save predictions for test set
    save_predictions(testfile, logistic_model, wv, output_file="test.predicted.embeddings.logistic.csv")

    # End timing the whole process
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total running time: {total_time:.2f} seconds")

    print("\nLogistic Regression model trained and predictions saved:")
    print("- test.predicted.embeddings.logistic.csv")
