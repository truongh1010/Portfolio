from collections import Counter
import sys
import time
import pandas as pd
from svector import svector

# Count word frequencies in the training set
def count_word_frequencies(trainfile):
    word_counter = Counter()
    data = pd.read_csv(trainfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        word_counter.update(words.split())
    return word_counter

# Prune low-frequency words based on min_count threshold
def read_from_with_pruning(textfile, word_counter, min_count=2):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        # Filter out words that appear less than min_count times in the training set
        filtered_words = [word for word in words.split() if word_counter[word] >= min_count]
        yield (1 if label == "+" else -1, filtered_words)

# Function to create a feature vector
def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    v['<bias>'] = 1
    return v

# Pruned version of the test function
def test_with_pruning(devfile, model, word_counter, min_count):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from_with_pruning(devfile, word_counter, min_count), 1):  # using the pruned read function
        err += label * (model.dot(make_vector(words))) <= 0
    return err / i  # i is |D| now

# Efficient averaged perceptron with pruning
def train_avg_perceptron_with_pruning(trainfile, devfile, epochs=10, min_count=2):
    t = time.time()
    # Count word frequencies in the training set
    word_counter = count_word_frequencies(trainfile)

    best_err = 1.0
    model = svector()  # This is w (weight vector)
    summed_model = svector()  # This is w_s (summed weights vector)

    # Training loop for multiple epochs
    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from_with_pruning(trainfile, word_counter, min_count), 1):
            sent = make_vector(words)
            
            # Predict and check if misclassified
            if label * (model.dot(sent)) <= 0:  # Misclassified
                updates += 1
                # Update weight vector w
                model += label * sent
            
            # Accumulate the current weight vector w into summed_model (w_s)
            summed_model += model
        
        # Evaluate dev error rate using the summed model (averaged model) and pruning
        averaged_model = divide_svector(summed_model, i)  # Average over all instances seen
        dev_err = test_with_pruning(devfile, averaged_model, word_counter, min_count)  # Use pruned test function for dev error
        
        # Track the best dev error and model
        if dev_err < best_err:
            best_err = dev_err
        
        print(f"Epoch {it}, updates {updates / i * 100:.1f}%, dev error {dev_err * 100:.1f}%")
    
    # Return the final averaged model
    final_model = divide_svector(summed_model, i)  # Final averaged model
    print(f"Best dev error {best_err * 100:.1f}%, |w|={len(final_model)}, time: {time.time() - t:.1f} secs")
    
    return final_model

# Helper function to divide each element of svector by a divisor
def divide_svector(vec, divisor):
    """Manually divide each element of svector by divisor."""
    result = svector()  # Create a new svector to store the result
    for key, value in vec.items():
        result[key] = value / divisor
    return result


if __name__ == "__main__":
    # ============= Vanila Perceptron Training================
    # best_model = train(sys.argv[1], sys.argv[2], 10)
    # ============= Average Perceptron Training================
    # best_model = train_avg_perceptron(sys.argv[1], sys.argv[2], 10)
    trainfile = sys.argv[1]
    devfile = sys.argv[2]

    # min_count to prune one-count or two-count words
    min_count = 3  # 2 to prune one-count words / 3 to prune two-count words

    best_model = train_avg_perceptron_with_pruning(trainfile, devfile, epochs=10, min_count=min_count)

    # Predict on the test set and save predictions
    test_data = pd.read_csv('test.csv')

    pred = []

    # Loop through each row in the test data
    for i, example in test_data.iterrows():
        id = example['id']  # Get the ID
        sentence = example['sentence']  # Get the sentence text
    
        # Create a vector from the sentence
        sent_vector = make_vector(sentence.split())
    
        # Predict sentiment using the model's dot product
        predicted_label = '+' if best_model.dot(sent_vector) > 0 else '-'
    
        # Append the prediction to the 'target' column in the DataFrame
        pred.append(predicted_label)

    # Add the predictions as the 'target' column in the test_data DataFrame
    test_data['target'] = pred

    # Save predictions to CSV
    test_data.to_csv('test.predicted.csv', index=False)
    print(f"Predictions saved to test.predicted.csv")
