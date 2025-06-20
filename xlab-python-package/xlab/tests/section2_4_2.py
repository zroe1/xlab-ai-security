"""
Tests for section 2.4.2 of the AI Security course.
"""

# import pprint
import pickle
import torch
import os

def get_100_examples():
    # Get the directory where this file is located
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data', 'cifar10_data.pkl')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        x_test_loaded = data['x_test']
        y_test_loaded = data['y_test']

    return x_test_loaded, y_test_loaded

def task1(model):
    x, y = get_100_examples()

    logits = model(x)
    predictions = torch.argmax(logits, axis=1)

    correct_predictions = predictions == y
    accuracy = torch.mean(correct_predictions.float())

    assert accuracy > 0.9, """
        Accuracy is less than 0.9. This indicates the model you passed in is not 
        correct. Try loaded that model by running `load_model(model_name='Standard', 
        threat_model='Linf')`
    """

    return accuracy