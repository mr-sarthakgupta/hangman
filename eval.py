import pickle
import os
import re
import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, accuracy_score, classification_report
import xgboost as xgb
import string
import ast

print(string.ascii_lowercase)
exit()

def return_word(name):
    match = re.search(r'~ ([a-zA-Z]+)\.', name)
    return match.group(1)

X = []
y = []

t = 0

for file_name in os.listdir('embeddings_val'):
    if file_name.endswith(".txt"):
        data_dict = {}
        with open(f"embeddings_val/{file_name}", 'r') as file:
            for line in file:
                # Strip whitespace and split the line into key-value pairs
                if ": " in line:
                    key, value = line.strip().split(": ", 1)
                    data_dict[key] = ast.literal_eval(value)
        sample = []
        for i in range(26):
            if string.ascii_lowercase[i] in list(data_dict.keys()):
                assert len(data_dict[string.ascii_lowercase[i]]) == 7
                sample.append(data_dict[string.ascii_lowercase[i]])
            else:
                sample.append([-1000, -1000, -1000, -1000, -1000, -1000, -1000])
        X.append(sample)
        y.append([1 if string.ascii_lowercase[i] in return_word(file_name) else 0 for i in range(26)])
        if t == 100000:
            break
        t += 1

X = np.array(X).sum(axis=2)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_test_bin = mlb.transform(y_test)


n_classes = y_train.shape[1]

def predict_multilabel(models, X_test, n_classes, threshold=0.5):
    # Predict probabilities for each label
    preds_proba = np.column_stack([
        model.predict(xgb.DMatrix(X_test)) for model in models
    ])
    
    # Convert probabilities to binary predictions
    preds_bin = (preds_proba >= threshold).astype(int)
    
    return preds_bin

models = pickle.load(open('xgboost_1014319.pkl', "rb"))

# Make predictions
y_pred = predict_multilabel(models, X_test, n_classes)

print(y_test.shape, y_pred.shape)

# Evaluate the model
def evaluate_multilabel_model(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    print("Hamming Loss:", hamming_loss(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Perform evaluation
evaluate_multilabel_model(y_test, y_pred)