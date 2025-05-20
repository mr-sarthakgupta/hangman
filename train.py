import pickle
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, accuracy_score, classification_report
import xgboost as xgb
import string
import ast


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
        
        print(data_dict)

        sample = []
        for i in range(26):
            if string.ascii_lowercase[i] in list(data_dict.keys()):
                assert len(data_dict[string.ascii_lowercase[i]]) == 6
                sample.append(data_dict[string.ascii_lowercase[i]])
            else:
                sample.append([-1000, -1000, -1000, -1000, -1000, -1000])
        X.append(sample)
        print(file_name)
        y.append([1 if string.ascii_lowercase[i] in return_word(file_name) else 0 for i in range(26)])
        t += 1
        

X = np.array(X).sum(axis=2)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
# y_train_bin = mlb.fit_transform(y_train)
# y_test_bin = mlb.transform(y_test)
y_train_bin = y_train
y_test_bin = y_test

print(y_train_bin.shape, y_train.shape)






# Train XGBoost model with multilabel support
def train_xgboost_multilabel(X_train, y_train_bin, n_classes):
    # Prepare models for each label
    models = []
    for i in range(n_classes):
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train_bin[:, i])
        
        # Set XGBoost parameters
        params = {
            'objective': 'binary:logistic',  # Binary logistic for each label
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 3
        }
        
        # Train model for this label
        model = xgb.train(params, dtrain, num_boost_round=100)
        models.append(model)
    
    return models

# Train the multilabel models
n_classes = y_train_bin.shape[1]

models = train_xgboost_multilabel(X_train, y_train_bin, n_classes)

# Prediction function
def predict_multilabel(models, X_test, n_classes, threshold=0.5):
    # Predict probabilities for each label
    preds_proba = np.column_stack([
        model.predict(xgb.DMatrix(X_test)) for model in models
    ])
    
    # Convert probabilities to binary predictions
    preds_bin = (preds_proba >= threshold).astype(int)
    
    return preds_bin

# Make predictions
y_pred = predict_multilabel(models, X_test, n_classes)

# Evaluate the model
def evaluate_multilabel_model(y_true, y_pred):
    print("Hamming Loss:", hamming_loss(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Perform evaluation
evaluate_multilabel_model(y_test_bin, y_pred)


print(y_test_bin.shape, y_pred.shape, 'meow')
# exit()
# Save the XGBoost models
# for i, model in enumerate(models):
#     model.save_model(f"xgboost_model_label_{i}.json")

dtrain = xgb.DMatrix(X_train, label=y_train)
        
# Train model
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1
}
# model = xgb.train(params, dtrain, num_boost_round=100)

# Save model with feature names

# pickle.dump(models, open(f'xgboost_{t}.pkl', 'wb'))

loaded_models = pickle.load(open('xgboost_1014319.pkl', "rb"))

# model.save_model('xgboost.json')

# # Load model with feature names
# loaded_model = xgb.Booster()
# loaded_model.load_model('xgboost.json')

load_pred = predict_multilabel(loaded_models, X_test, n_classes)

print(y_test_bin.shape, load_pred.shape, 'meow 2')

evaluate_multilabel_model(y_test_bin, load_pred)

# # Optional: Predict for new data
# def predict_new_samples(models, X_new, threshold=0.5):
#     preds_proba = np.column_stack([
#         model.predict(xgb.DMatrix(X_new)) for model in models
#     ])
#     return (preds_proba >= threshold).astype(int)

# # Example of how to use the trained model
# X_new = create_multilabel_dataset(n_samples=500, n_features=7, n_classes=26, n_labels=12)[0]
# new_predictions = predict_new_samples(models, X_new)