#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
import pandas as pd
from svector import svector


def read_from(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        yield (1 if label=="+" else -1, words.split())


def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    v['<bias>'] = 1
    return v
    

def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now
   

def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()       # weight vector
    best_model = None       # this store the best model

    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
        dev_err = test(devfile, model)
        
        # check for best model
        if dev_err < best_err:
            best_err = dev_err
            best_model = model.copy()       # update best model

        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

    return best_model



def train_avg_perceptron(trainfile, devfile, epochs=10):
    t = time.time()
    best_err = 1.0
    model = svector()  # This is w (weight vector)
    avg_model = svector()  # This is w_a (cumulative weight vector)
    c = 0  # The counter

    # Training loop for multiple epochs
    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent = make_vector(words)
            
            # Predict and check if misclassified
            if label * (model.dot(sent)) <= 0:  # Misclassified
                updates += 1
                # Update weight vector w
                model += label * sent
                
                # Update cumulative weight vector w_a
            avg_model += c * label * sent
            
            # Increment the counter
            c += 1
        
        # Check dev error rate after each epoch
        dev_err = test(devfile, model)
        
        if dev_err < best_err:
            best_err = dev_err
            # best_model = model.copy()  # Save the best performing model
        
        print(f"Epoch {it}, updates {updates / i * 100:.1f}%, dev error {dev_err * 100:.1f}%")
    
    # Return the final averaged model: c * model - avg_model
    final_model = (c * model) - avg_model
    print(f"Best dev error {best_err * 100:.1f}%, |w|={len(final_model)}, time: {time.time() - t:.1f} secs")
    
    return final_model

def divide_svector(vec, divisor):
    """Manually divide each element of svector by divisor."""
    result = svector()  # Create a new svector to store the result
    for key, value in vec.items():
        result[key] = value / divisor
    return result


"""
def train_avg_perceptron_efficient(trainfile, devfile, epochs=10):
    t = time.time()
    best_err = 1.0
    model = svector()  # This is w (weight vector)
    summed_model = svector()  # This is w_s (summed weights vector)

    # Training loop for multiple epochs
    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent = make_vector(words)
            
            # Predict and check if misclassified
            if label * (model.dot(sent)) <= 0:  # Misclassified
                updates += 1
                # Update weight vector w
                model += label * sent
            
            # Accumulate the current weight vector w into summed_model (w_s)
            summed_model += model
        
        # Evaluate dev error rate using the summed model (averaged model)
        averaged_model = divide_svector(summed_model, i)  # Average over all instances seen
        dev_err = test(devfile, averaged_model)  # Use averaged model for dev error
        
        # Track the best dev error and model
        if dev_err < best_err:
            best_err = dev_err
        
        print(f"Epoch {it}, updates {updates / i * 100:.1f}%, dev error {dev_err * 100:.1f}%")
    
    # Return the final averaged model
    final_model = divide_svector(summed_model, i)  # Final averaged model
    print(f"Best dev error {best_err * 100:.1f}%, |w|={len(final_model)}, time: {time.time() - t:.1f} secs")
    
    return final_model

"""

def train_avg_perceptron_efficient(trainfile, devfile, epochs=10):
    t = time.time()
    best_err = 1.0
    model = svector()  # This is w (weight vector)
    summed_model = svector()  # This will be w_s (summed weights vector)
    c = 1  # Start counter from 1 to weight the model updates

    # Training loop for multiple epochs
    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent = make_vector(words)
            
            # Predict and check if misclassified
            if label * (model.dot(sent)) <= 0:  # Misclassified
                updates += 1
                # Update weight vector w
                model += label * sent
            
            # Accumulate model into summed_model without multiplying by c every time
            summed_model += model
            
            # Increment the counter for every instance
            c += 1
        
        # Evaluate dev error rate using the summed model (averaged model)
        averaged_model = divide_svector(summed_model, c - 1)  # Average over all instances seen
        dev_err = test(devfile, averaged_model)  # Use averaged model for dev error
        
        # Track the best dev error and model
        if dev_err < best_err:
            best_err = dev_err
        
        print(f"Epoch {it}, updates {updates / i * 100:.1f}%, dev error {dev_err * 100:.1f}%")
    
    # Return the final averaged model
    final_model = divide_svector(summed_model, c - 1)  # Final averaged model
    print(f"Best dev error {best_err * 100:.1f}%, |w|={len(final_model)}, time: {time.time() - t:.1f} secs")
    
    return final_model





def misclassify_in_dev(devfile, model, n=5):
    incorrect_pos = []  # List to store negative examples predicted as positive
    incorrect_neg = []  # List to store positive examples predicted as negative

    # Read and process the dev set
    for label, words in read_from(devfile):
        sent_vector = make_vector(words)  
        pred = model.dot(sent_vector)  # Dot product gives prediction 
        
        # If it's a negative example but predicted as positive
        if label == -1 and pred > 0:
            incorrect_pos.append((words, pred))  
        
        # If it's a positive example but predicted as negative
        if label == 1 and pred < 0:
            incorrect_neg.append((words, pred))  

    # Sort both lists by the confidence of the incorrect prediction
    incorrect_pos.sort(key=lambda x: x[1], reverse=True)  # Sort by high confidence for positive prediction
    incorrect_neg.sort(key=lambda x: x[1])  # Sort by low confidence for negative prediction

    # Select top N most strongly misclassified examples
    return incorrect_pos[:n], incorrect_neg[:n]






if __name__ == "__main__":
    # ============= Vanila Perceptron Training================
    # best_model = train(sys.argv[1], sys.argv[2], 10)
    # ============= Average Perceptron Training================
    # best_model = train_avg_perceptron(sys.argv[1], sys.argv[2], 10)

    best_model = train_avg_perceptron(sys.argv[1], sys.argv[2], 10)

    

    # # Get the top 20 most positive and top 20 most negative features
    # # Sort the features (words) by their weights in descending order
    # sorted_features = sorted(best_model.items(), key=lambda item: item[1], reverse=True)
    # # Get the top 20 most positive features
    # top_positive = sorted_features[:20]
    # # Get the top 20 most negative features (reverse order of sorting)
    # top_negative = sorted_features[-20:]
    # # Print the top positive and negative features
    # print("Top 20 Most Positive Features:")
    # for feature, weight in top_positive:
    #     print(f"{feature}: {weight:.4f}")
    # print("\nTop 20 Most Negative Features:")
    # for feature, weight in top_negative:
    #     print(f"{feature}: {weight:.4f}")

    # incorrect_pos, incorrect_neg = misclassify_in_dev(sys.argv[2], best_model, 5)
    # # Print the results
    # print("Top 5 Negative Examples Predicted as Positive:")
    # for words, pred in incorrect_pos:
    #     print(f"Example: {' '.join(words)} | Prediction: {pred:.4f}")

    # print("\nTop 5 Positive Examples Predicted as Negative:")
    # for words, pred in incorrect_neg:
    #     print(f"Example: {' '.join(words)} | Prediction: {pred:.4f}")


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

    # Convert to a DataFrame and save to CSV
    # df = pd.DataFrame(pred)
    test_data.to_csv('test.predicted11111.csv', index=False)
    print(f"Predictions saved to test.predicted.csv")

