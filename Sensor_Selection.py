import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class SensorSelectionNN(nn.Module):
    def __init__(self, input_dim):
        super(SensorSelectionNN, self).__init__()
        # One-to-One Layer with L1 regularization
        self.one_to_one = nn.Linear(input_dim, input_dim, bias=False)
        # Fully connected layers
        self.hidden1 = nn.Linear(input_dim, 400)
        self.hidden2 = nn.Linear(400, 100)
        self.output = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.one_to_one(x)  # Apply one-to-one weighting
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.sigmoid(self.output(x))
        return x

def train_nn(csv_file, feature_columns, label_column, l1_penalty=0.01, epochs=100, batch_size=32, learning_rate=0.001):
    # Load dataset
    data = pd.read_csv(csv_file)
    features = data[feature_columns].values
    labels = data[label_column].values

    # Normalize features
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    # Convert to PyTorch tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Define model, loss, and optimizer
    input_dim = len(feature_columns)
    model = SensorSelectionNN(input_dim)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X[indices], y[indices]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Add L1 regularization to the one-to-one layer
            l1_loss = torch.norm(model.one_to_one.weight, p=1)
            loss += l1_penalty * l1_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Extract feature importance from one-to-one layer
    feature_weights = model.one_to_one.weight.detach().cpu().numpy().flatten()
    selected_features = [feature_columns[i] for i, weight in enumerate(feature_weights) if weight != 0]

    print("\nSelected Features:")
    for feature in selected_features:
        print(feature)

    return model, selected_features

# Example Usage
if __name__ == "__main__":
    # Path to the CSV file
    csv_file = "example.csv"  # Replace with your file path

    # Define feature and label columns
    feature_columns = ["Column1", "Column2", "Column3", "Column4"]  # Replace with your sensor columns
    label_column = "label"  # Replace with your event column

    # Train the Neural Network
    model, selected_features = train_nn(
        csv_file, 
        feature_columns, 
        label_column, 
        l1_penalty=0.01, 
        epochs=100, 
        batch_size=32, 
        learning_rate=0.001
    )
