from sklearn.model_selection import train_test_split
import os
import string
import ast
import re
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer

class MulticlassHammingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim[0] * input_dim[1], num_classes)
        self.linear.requires_grad = True
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.linear(x))
    
    # def train(self):
    #     self.linear.re

def hamming_loss(y_true, y_pred, threshold=0.5):
    """
    Calculate Hamming loss for multilabel classification
    
    Args:
    y_true (torch.Tensor): True binary labels
    y_pred (torch.Tensor): Predicted probabilities
    threshold (float): Threshold for converting probabilities to binary predictions
    
    Returns:
    torch.Tensor: Hamming loss
    """
    y_pred_binary = (y_pred > threshold).float()
    incorrect_predictions = torch.abs(y_true - y_pred_binary)
    return torch.mean(incorrect_predictions)

def train_multiclass_hamming_classifier(X_train, y_train, X_val=None, y_val=None, 
                                         input_dim=(26, 6), num_classes=26, 
                                         learning_rate=0.01, epochs=100, 
                                         batch_size=32, device='cpu'):
    """
    Train a multiclass classification model with Hamming loss
    
    Args:
    X_train (np.ndarray): Training input data of shape (n_samples, 26, 6)
    y_train (np.ndarray): Training labels of shape (n_samples, num_classes)
    X_val (np.ndarray, optional): Validation input data
    y_val (np.ndarray, optional): Validation labels
    
    Returns:
    dict: Training results including trained model, training history
    """
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    # Prepare validation data if provided
    if X_val is not None and y_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
    
    # Move to specified device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    # Initialize model, loss, and optimizer
    model = MulticlassHammingClassifier(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history tracking
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        epoch_loss = 0

        print('meow', model.linear.weight, model.linear.bias)
        
        for batch_x, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_x)
            loss = hamming_loss(batch_y, outputs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(dataloader)
        train_losses.append(train_loss)
        
        # Validation phase
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = hamming_loss(y_val, val_outputs)
                val_losses.append(val_loss.item())
        
        # Optional: Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}')

        # Save the model with the best validation loss
        if X_val is not None and y_val is not None:
            if epoch == 0 or val_loss.item() < min(val_losses):
                torch.save(model.state_dict(), f'models/model_{min(val_losses)}.pth')
        
        
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def return_word(name):
    match = re.search(r'~ ([a-zA-Z]+)\.', name)
    return match.group(1)

# Example usage
if __name__ == '__main__':

    X = []
    y = []

    t = 0

    for file_name in os.listdir('embeddings_train/'):
        if file_name.endswith(".txt"):
            data_dict = {}
            with open(f"embeddings_train/{file_name}", 'r') as file:
                for line in file:
                    # Strip whitespace and split the line into key-value pairs
                    if ": " in line:
                        key, value = line.strip().split(": ", 1)
                        data_dict[key] = ast.literal_eval(value)
            
            sample = []
            for i in range(26):
                if string.ascii_lowercase[i] in list(data_dict.keys()):
                    assert len(data_dict[string.ascii_lowercase[i]]) == 6
                    sample.append(data_dict[string.ascii_lowercase[i]])
                else:
                    sample.append([-1000, -1000, -1000, -1000, -1000, -1000])
            X.append(sample)
            # print(file_name)
            y.append([1 if string.ascii_lowercase[i] in return_word(file_name) else 0 for i in range(26)])
            t += 1

    X = np.array(X)
    y = np.array(y)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Generate random example data
    
    # Train the model
    results = train_multiclass_hamming_classifier(X_train, y_train, X_val, y_val)