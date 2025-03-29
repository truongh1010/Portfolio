#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from collections import Counter
from svector import svector  # For one-hot perceptron

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
    return {word for word, count in word_counts.items() if count > 1}

# Generate dense vector (Word2Vec with bias)
def make_dense_vector_with_bias(words, wv, valid_words):
    vectors = [wv[word] for word in words if word in valid_words and word in wv]
    avg_vector = np.mean(vectors, axis=0) if vectors else np.zeros(wv.vector_size)
    return np.append(avg_vector, 1)  # Add bias

# Generate sparse vector (One-Hot)
def make_sparse_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    v['<bias>'] = 1
    return v

# Train Word2Vec-based Averaged Perceptron
def train_word2vec_perceptron(trainfile, wv, valid_words, epochs=10):
    model = np.zeros(wv.vector_size + 1)
    avg_model = np.zeros(wv.vector_size + 1)
    c = 0

    for it in range(1, epochs + 1):
        for label, words in read_from(trainfile):
            sent_vector = make_dense_vector_with_bias(words, wv, valid_words)
            if label * np.dot(model, sent_vector) <= 0:  # Misclassified
                model += label * sent_vector
            avg_model += model
            c += 1
    return avg_model / c

# Train One-Hot Averaged Perceptron
def train_one_hot_perceptron(trainfile, epochs=10):
    model = svector()
    avg_model = svector()
    c = 0

    for it in range(1, epochs + 1):
        for label, words in read_from(trainfile):
            sent_vector = make_sparse_vector(words)
            if label * model.dot(sent_vector) <= 0:  # Misclassified
                model += label * sent_vector
            avg_model += model
            c += 1
    return avg_model * (1 / c)

# Predict with Word2Vec-based Averaged Perceptron
def predict_with_word2vec(devfile, model, wv, valid_words):
    predictions = []
    for label, words in read_from(devfile):
        sent_vector = make_dense_vector_with_bias(words, wv, valid_words)
        pred = '+' if np.dot(model, sent_vector) > 0 else '-'
        predictions.append((label, pred, words))
    return predictions

# Predict with One-Hot Averaged Perceptron
def predict_with_one_hot(devfile, model):
    predictions = []
    for label, words in read_from(devfile):
        sent_vector = make_sparse_vector(words)
        pred = '+' if model.dot(sent_vector) > 0 else '-'
        predictions.append((label, pred, words))
    return predictions

# Compare predictions and find mismatches
def compare_predictions(word2vec_preds, one_hot_preds):
    mismatches = []
    for (label1, pred1, words1), (label2, pred2, words2) in zip(word2vec_preds, one_hot_preds):
        if label1 == 1 and pred1 == '+' and pred2 == '-':  # Word2Vec correct, One-Hot wrong
            mismatches.append((label1, words1, pred1, pred2))
    return mismatches

# Main entry point
if __name__ == "__main__":
    embedding_file = "embs_train.kv"  # Replace with your embedding file path
    trainfile = sys.argv[1]
    devfile = sys.argv[2]

    # Load embeddings
    wv = load_embeddings(embedding_file)

    # Prune one-count words
    valid_words = prune_one_count_words(trainfile)

    # Train Word2Vec-based Averaged Perceptron
    print("Training Word2Vec-based Averaged Perceptron...")
    word2vec_model = train_word2vec_perceptron(trainfile, wv, valid_words)
    print("Word2Vec-based model training complete.")

    # Train One-Hot Averaged Perceptron
    print("Training One-Hot Averaged Perceptron...")
    one_hot_model = train_one_hot_perceptron(trainfile)
    print("One-Hot model training complete.")

    # Generate predictions
    word2vec_preds = predict_with_word2vec(devfile, word2vec_model, wv, valid_words)
    one_hot_preds = predict_with_one_hot(devfile, one_hot_model)

    # Compare predictions
    mismatches = compare_predictions(word2vec_preds, one_hot_preds)

    # Print mismatched examples
    print("Examples where Word2Vec is correct and One-Hot is wrong:")
    for label, words, word2vec_pred, one_hot_pred in mismatches[:2]:  # Show first 2 examples
        print(f"Sentence: {' '.join(words)}")
        print(f"True Label: {'+' if label == 1 else '-'}")
        print(f"Word2Vec Prediction: {word2vec_pred}")
        print(f"One-Hot Prediction: {one_hot_pred}")
        print("-" * 50)
