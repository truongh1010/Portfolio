# #!/usr/bin/env python3

# import sys
# import numpy as np
# import pandas as pd
# from gensim.models import KeyedVectors

# # Load word embeddings
# def load_embeddings(embedding_file):
#     return KeyedVectors.load(embedding_file)

# # Read data
# def read_from(textfile):
#     data = pd.read_csv(textfile)
#     for i in range(len(data)):
#         id, words, label = data.iloc[i]
#         yield (1 if label == "+" else -1, words.split())

# # Generate dense vector (average word embeddings)
# def make_dense_vector(words, wv):
#     vectors = [wv[word] for word in words if word in wv]
#     if not vectors:
#         return np.zeros(wv.vector_size)  # Return a zero vector if no valid embeddings
#     return np.mean(vectors, axis=0)

# # Test perceptron with dense embeddings
# def test_dense(devfile, model, wv):
#     tot, err = 0, 0
#     for i, (label, words) in enumerate(read_from(devfile), 1):
#         sent_vector = make_dense_vector(words, wv)  # Get dense vector
#         prediction = np.dot(model, sent_vector)  # Compute dot product
#         err += label * prediction <= 0  # Count errors
#     return err / i  # i is |D| now

# # Basic Perceptron with dense embeddings
# def train_basic_perceptron_with_embeddings(trainfile, devfile, wv, epochs=5):
#     # Initialize model as a NumPy array with the same size as word embeddings
#     model = np.zeros(wv.vector_size)
#     best_model = None
#     best_err = 1.0

#     for it in range(1, epochs + 1):
#         updates = 0
#         for i, (label, words) in enumerate(read_from(trainfile), 1):
#             sent_vector = make_dense_vector(words, wv)  # Get dense vector
#             prediction = np.dot(model, sent_vector)  # Compute dot product
#             if label * prediction <= 0:  # Misclassified
#                 updates += 1
#                 model += label * sent_vector  # Update weights
#         # Test the model on the development set
#         dev_err = test_dense(devfile, model, wv)
#         if dev_err < best_err:
#             best_err = dev_err
#             best_model = model.copy()  # Save the best model
#         print(f"Epoch {it}, updates {updates / i * 100:.1f}%, dev error {dev_err * 100:.1f}%")
#     print(f"Best dev error {best_err * 100:.1f}%")
#     return best_model

# # Main entry point
# if __name__ == "__main__":
#     embedding_file = "embs_train.kv"  # Replace with your embedding file path
#     trainfile = sys.argv[1]
#     devfile = sys.argv[2]

#     # Load embeddings
#     wv = load_embeddings(embedding_file)

#     # Train and test perceptron with embeddings
#     best_model = train_basic_perceptron_with_embeddings(trainfile, devfile, wv, epochs=5)

#     # Save predictions for Kaggle submission
#     test_data = pd.read_csv('test.csv')
#     predictions = []
#     for i, row in test_data.iterrows():
#         sent_vector = make_dense_vector(row['sentence'].split(), wv)
#         predicted_label = '+' if np.dot(best_model, sent_vector) > 0 else '-'
#         predictions.append(predicted_label)
#     test_data['target'] = predictions
#     test_data.to_csv('test.predicted.csv', index=False)
#     print(f"Predictions saved to test.predicted.csv")


#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

# Load word embeddings
def load_embeddings(embedding_file):
    return KeyedVectors.load(embedding_file)

# Read data
def read_from(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        yield (1 if label == "+" else -1, words.split())

# Generate dense vector (average word embeddings with a bias term)
def make_dense_vector_with_bias(words, wv):
    vectors = [wv[word] for word in words if word in wv]
    if not vectors:
        avg_vector = np.zeros(wv.vector_size)
    else:
        avg_vector = np.mean(vectors, axis=0)
    return np.append(avg_vector, 1)  # Add bias as the last dimension

# Test perceptron with dense embeddings and a bias term
def test_dense_with_bias(devfile, model, wv):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1):
        sent_vector = make_dense_vector_with_bias(words, wv)  # Get dense vector with bias
        prediction = np.dot(model, sent_vector)  # Compute dot product
        err += label * prediction <= 0  # Count errors
    return err / i  # i is |D| now

# Basic Perceptron with dense embeddings and a bias term
def train_basic_perceptron_with_bias(trainfile, devfile, wv, epochs=5):
    # Initialize model as a NumPy array with size of embeddings + 1 for the bias term
    model = np.zeros(wv.vector_size + 1)
    best_model = None
    best_err = 1.0

    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent_vector = make_dense_vector_with_bias(words, wv)  # Get dense vector with bias
            prediction = np.dot(model, sent_vector)  # Compute dot product
            if label * prediction <= 0:  # Misclassified
                updates += 1
                model += label * sent_vector  # Update weights
        # Test the model on the development set
        dev_err = test_dense_with_bias(devfile, model, wv)
        if dev_err < best_err:
            best_err = dev_err
            best_model = model.copy()  # Save the best model
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

    # Train and test perceptron with embeddings and bias
    best_model = train_basic_perceptron_with_bias(trainfile, devfile, wv, epochs=10)

    # Save predictions for Kaggle submission
    test_data = pd.read_csv('test.csv')
    predictions = []
    for i, row in test_data.iterrows():
        sent_vector = make_dense_vector_with_bias(row['sentence'].split(), wv)
        predicted_label = '+' if np.dot(best_model, sent_vector) > 0 else '-'
        predictions.append(predicted_label)
    test_data['target'] = predictions
    test_data.to_csv('test.predicted.csv', index=False)
    print(f"Predictions saved to test.predicted.csv")
