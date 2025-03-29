"""
import sys
import time
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# First pass: Count word frequencies in the training set
def count_word_frequencies(trainfile):
    word_counter = Counter()
    data = pd.read_csv(trainfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        word_counter.update(words.split())
    return word_counter

# Second pass: Prune low-frequency words based on min_count threshold
def read_from_with_pruning(textfile, word_counter, min_count=2):
    data = pd.read_csv(textfile)
    sentences = []
    labels = []
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        # Filter out words that appear less than min_count times in the training set
        filtered_words = [word for word in words.split() if word_counter[word] >= min_count]
        sentences.append(" ".join(filtered_words))
        labels.append(1 if label == "+" else 0)  # Using 1 for positive and 0 for negative
    return sentences, labels

# Negation handling function
def handle_negation(sentence):
    words = sentence.split()
    negation_words = {"not", "n't", "never", "no"}
    new_words = []
    negation_active = False

    for word in words:
        if word in negation_words:
            negation_active = True
            new_words.append(word)
        elif negation_active:
            new_words.append(f"NOT_{word}")  # Mark words following negation
            negation_active = False
        else:
            new_words.append(word)

    return " ".join(new_words)

# Model evaluation function with cross-validation
def evaluate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{model.__class__.__name__} - Cross-validated accuracy: {scores.mean() * 100:.2f}%")
    return scores.mean()

# Main function to evaluate SVC model
def main(trainfile, devfile, min_count=2):
    # Step 1: Count word frequencies in the training set for pruning
    word_counter = count_word_frequencies(trainfile)
    
    # Step 2: Read and prune training and dev data
    train_sentences, train_labels = read_from_with_pruning(trainfile, word_counter, min_count)
    dev_sentences, dev_labels = read_from_with_pruning(devfile, word_counter, min_count)
    
    # Step 3: Apply negation handling
    train_sentences = [handle_negation(sentence) for sentence in train_sentences]
    dev_sentences = [handle_negation(sentence) for sentence in dev_sentences]
    
    # Step 4: Use TF-IDF with bigrams for feature extraction
    vectorizer = TfidfVectorizer(min_df=min_count, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_sentences)
    X_dev = vectorizer.transform(dev_sentences)

    # Initialize SVC model
    svc_model = SVC(kernel='linear', C=1)
    
    # Evaluate the SVC model
    print("\nEvaluating SVC model...")
    start_time = time.time()
    
    # Cross-validation score
    cv_score = evaluate_model(svc_model, X_train, train_labels)
    
    # Fit on training data
    svc_model.fit(X_train, train_labels)
    
    # Calculate dev error rate
    dev_accuracy = svc_model.score(X_dev, dev_labels)
    dev_error_rate = 1 - dev_accuracy
    elapsed_time = time.time() - start_time  # Calculate elapsed time for SVC model

    # Display results
    print(f"SVC - Dev Error Rate: {dev_error_rate * 100:.2f}%")
    print(f"SVC - Running Time: {elapsed_time:.2f} seconds")
    print(f"SVC - Cross-validated Accuracy: {cv_score * 100:.2f}%")

    # Return model and vectorizer for possible test predictions
    return svc_model, vectorizer


# Run the script
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <trainfile> <devfile>")
        sys.exit(1)

    trainfile = sys.argv[1]
    devfile = sys.argv[2]

    # Set minimum count for word pruning
    min_count = 2
    best_model, vectorizer = main(trainfile, devfile, min_count=min_count)

    # Optional: Generate predictions on the test set if you have a test.csv file
    test_data = pd.read_csv('test.csv')
    X_test = vectorizer.transform(test_data['sentence'])  # Transform test sentences into features
    predictions = best_model.predict(X_test)

    # Save predictions in the required format
    # test_data['target'] = ['+' if pred == 1 else '-' for pred in predictions]
    # test_data.to_csv('test.predicted.csv', index=False)
    # print("Predictions saved to test.predicted.csv")
"""


import sys  # Import sys module to handle command-line arguments
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Enhanced negation handling with "NOT_" prefix
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

# First pass: Count word frequencies in the training set
def count_word_frequencies(trainfile):
    word_counter = Counter()
    data = pd.read_csv(trainfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        word_counter.update(words.split())
    return word_counter

# Second pass: Prune low-frequency words based on min_count threshold
def preprocess_data_with_pruning(textfile, word_counter, min_count=2):
    data = pd.read_csv(textfile)
    sentences = []
    labels = []
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        # Prune low-frequency words
        filtered_words = [word for word in words.split() if word_counter[word] >= min_count]
        sentence = " ".join(filtered_words)
        
        # Apply negation handling
        sentence = handle_negation(sentence)
        
        # Add sentence to list
        sentences.append(sentence)
        labels.append(1 if label == "+" else 0)
    return sentences, labels

# Main function to run logistic regression with TF-IDF
def train_with_enhanced_features(trainfile, devfile, min_count=2):
    # Count word frequencies for pruning
    word_counter = count_word_frequencies(trainfile)
    
    # Preprocess training and dev data
    train_sentences, train_labels = preprocess_data_with_pruning(trainfile, word_counter, min_count)
    dev_sentences, dev_labels = preprocess_data_with_pruning(devfile, word_counter, min_count)

    # Use TF-IDF with bigrams and trigrams
    vectorizer = TfidfVectorizer(min_df=min_count, ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(train_sentences)
    X_dev = vectorizer.transform(dev_sentences)

    # Initialize and train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    # Evaluate on dev set
    dev_accuracy = model.score(X_dev, dev_labels)
    dev_error_rate = 1 - dev_accuracy
    print(f"Dev Error Rate: {dev_error_rate * 100:.2f}%")

    return model, vectorizer

# Run script
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <trainfile> <devfile>")
        sys.exit(1)

    trainfile = sys.argv[1]
    devfile = sys.argv[2]

    # Set minimum count for word pruning
    min_count = 2
    best_model, vectorizer = train_with_enhanced_features(trainfile, devfile, min_count=min_count)

    # Optional: Generate predictions on the test set if you have a test.csv file
    test_data = pd.read_csv('test.csv')
    X_test = vectorizer.transform(test_data['sentence'])  # Transform test sentences into features
    predictions = best_model.predict(X_test)

    # Save predictions in the required format
    test_data['target'] = ['+' if pred == 1 else '-' for pred in predictions]
    test_data.to_csv('test.predicted.csv', index=False)
    print("Predictions saved to test.predicted.csv")
