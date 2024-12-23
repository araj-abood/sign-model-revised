
import torch
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scripts.train_model import SignLanguageModel


def generate_class_to_idx(data_dir):
    """
    Generate a class-to-index mapping from the directory structure.

    Parameters:
        data_dir (str): Path to the dataset directory.

    Returns:
        dict: A mapping of class names to integer indices.
    """
    classes = sorted(os.listdir(data_dir))  # Sort for consistent mapping
    return {cls: idx for idx, cls in enumerate(classes)}


class TestDataset:
    def __init__(self, data_dir, class_to_idx, max_landmark_size=126):
        self.samples = []
        self.labels = []
        self.max_landmark_size = max_landmark_size

        # Load samples
        for cls, idx in class_to_idx.items():
            cls_dir = os.path.join(data_dir, cls)
            for file_name in os.listdir(cls_dir):
                file_path = os.path.join(cls_dir, file_name)

                try:
                    with open(file_path, "r") as f:
                        landmarks = json.load(f)

                    # Flatten landmarks
                    flattened = []
                    for hand in landmarks:
                        flattened.extend([val for lm in hand for val in lm.values()])

                    # Pad or truncate
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

def evaluate_model(model_path, test_dir, class_to_idx):
    """
    Evaluate the trained model on a test dataset.

    Parameters:
        model_path (str): Path to the saved model file.
        test_dir (str): Path to the test dataset directory.
        class_to_idx (dict): Mapping of class names to indices.
    """
    dataset = TestDataset(test_dir, class_to_idx)
    inputs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])  # Ensure labels are tensors

    # Load model
    checkpoint = torch.load(model_path)
    model = SignLanguageModel(input_size=126, num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

    predictions = predictions.cpu().numpy()  
    labels = labels.cpu().numpy()            

    # Evaluate
    acc = accuracy_score(labels, predictions)
    print(f"Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=list(class_to_idx.keys())))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))

    
# Example usage
if __name__ == "__main__":
    model_path = "data/model/sign_language_model.pth"
    test_dir = "data/test_landmarks"
    class_to_idx = generate_class_to_idx(test_dir)


    evaluate_model(model_path, test_dir, class_to_idx)
