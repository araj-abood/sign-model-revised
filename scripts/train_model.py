import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

def plot_training_visualization(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = len(train_losses)
    x = range(1, epochs + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.plot(x, train_losses, label="Training Loss", marker="o")
    plt.plot(x, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    plt.plot(x, train_accuracies, label="Training Accuracy", marker="o")
    plt.plot(x, val_accuracies, label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


class LandmarkDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.max_landmark_size = 2 * 21 * 3  # Max size based on 21 hand landmarks, 3 coords each

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for file_name in os.listdir(cls_dir):
                file_path = os.path.join(cls_dir, file_name)

                try:
                    with open(file_path, "r") as f:
                        landmarks = json.load(f)

                    flattened = []
                    for hand in landmarks:
                        flattened.extend([val for lm in hand for val in lm.values()])

                    if len(flattened) < self.max_landmark_size:
                        flattened.extend([0] * (self.max_landmark_size - len(flattened)))
                    elif len(flattened) > self.max_landmark_size:
                        flattened = flattened[:self.max_landmark_size]

                    self.samples.append(flattened)
                    self.labels.append(self.class_to_idx[cls])

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), self.labels[idx]


class SignLanguageModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignLanguageModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


def train_model(train_dir, val_dir, model_save_path, epochs=50, batch_size=32, learning_rate=0.001):
    train_dataset = LandmarkDataset(train_dir)
    val_dataset = LandmarkDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = len(train_dataset[0][0])
    num_classes = len(train_dataset.classes)

    model = SignLanguageModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0

        for landmarks, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        train_accuracy = 100 * train_correct / len(train_dataset)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for landmarks, labels in val_loader:
                outputs = model(landmarks)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        val_accuracy = 100 * val_correct / len(val_dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_losses[-1]:.4f} | Train Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"  Val Loss: {val_losses[-1]:.4f} | Val Accuracy: {val_accuracies[-1]:.2f}%")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'num_classes': num_classes,
                'class_to_idx': train_dataset.class_to_idx
            }, model_save_path)
            print(f"  Best model saved to {model_save_path}")

    print("Training complete.")
    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":
    train_directory = "data/train_landmarks"
    val_directory = "data/val_landmarks"
    model_save_path = "data/model/sign_language_model.pth"

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        train_directory, val_directory, model_save_path
    )

    plot_training_visualization(train_losses, val_losses, train_accuracies, val_accuracies)

