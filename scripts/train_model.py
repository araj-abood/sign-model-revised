import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LandmarkDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.max_landmark_size = 2 * 21 * 3

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

def train_model(data_dir, model_save_path, epochs=50, batch_size=32, learning_rate=0.001):
    dataset = LandmarkDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = len(dataset[0][0])  
    num_classes = len(dataset.classes)

    model = SignLanguageModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for landmarks, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
        'class_to_idx': dataset.class_to_idx
    }, model_save_path) 
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    data_dir = "data/train_landmarks"  
    model_save_path = "data/model/sign_language_model.pth"  
    train_model(data_dir, model_save_path)
