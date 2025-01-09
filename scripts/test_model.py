import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from train_model import SignLanguageModel


def generate_class_to_idx(data_dir):
    classes = sorted(os.listdir(data_dir)) 
    return {cls: idx for idx, cls in enumerate(classes)}


class TestDataset:
    def __init__(self, data_dir, class_to_idx, max_landmark_size=126):
        self.samples = []
        self.labels = []
        self.max_landmark_size = max_landmark_size

        for cls, idx in class_to_idx.items():
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
                    self.labels.append(idx)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), self.labels[idx]


def plot_metrics(metrics):
    """
    Plots a bar chart for accuracy, precision, recall, and F1-score.
    """
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


def evaluate_model(model_path, test_dir, class_to_idx):
    dataset = TestDataset(test_dir, class_to_idx)
    inputs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])  

    checkpoint = torch.load(model_path)
    model = SignLanguageModel(input_size=126, num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

    predictions = predictions.cpu().numpy()  
    labels = labels.cpu().numpy()            

    # Metrics
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=list(class_to_idx.keys())))

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot metrics and confusion matrix
    plot_metrics(metrics)
    plot_confusion_matrix(cm, list(class_to_idx.keys()))


if __name__ == "__main__":
    model_path = "data/model/sign_language_model.pth"
    test_dir = "data/test_landmarks"
    class_to_idx = generate_class_to_idx(test_dir)

    evaluate_model(model_path, test_dir, class_to_idx)
