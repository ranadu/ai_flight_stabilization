import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


def load_data(filename="training_data.csv"):
    features = []
    targets = []

    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta = float(row["theta"])
            q = float(row["q"])
            u = float(row["u"])

            features.append([theta, q])
            targets.append([u])

    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    return X, y


def normalize_data(X, y):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8

    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-8

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    return X_norm, y_norm, X_mean, X_std, y_mean, y_std


def train_model():
    X, y = load_data("training_data.csv")
    X_norm, y_norm, X_mean, X_std, y_mean, y_std = normalize_data(X, y)

    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    model = ControlNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200

    for epoch in range(epochs):
        model.train()

        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs} | Loss = {loss.item():.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "X_mean": X_mean,
            "X_std": X_std,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "control_model.pth"
    )

    print("Model saved to control_model.pth")


if __name__ == "__main__":
    train_model()