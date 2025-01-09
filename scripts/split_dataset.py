import os
import shutil
import random

def split_dataset(input_dir, train_dir, val_dir, test_dir, val_ratio=0.1, test_ratio=0.2):
    """
    Splits the dataset into training, validation, and testing sets.

    Args:
        input_dir (str): Directory containing the input dataset.
        train_dir (str): Directory to store the training dataset.
        val_dir (str): Directory to store the validation dataset.
        test_dir (str): Directory to store the testing dataset.
        val_ratio (float): Fraction of data to use for validation.
        test_ratio (float): Fraction of data to use for testing.
    """

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        train_class_path = os.path.join(train_dir, class_name)
        val_class_path = os.path.join(val_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        files = os.listdir(class_path)
        random.shuffle(files)

        test_split_idx = int(len(files) * (1 - test_ratio))
        val_split_idx = int(test_split_idx * (1 - val_ratio))

        train_files = files[:val_split_idx]
        val_files = files[val_split_idx:test_split_idx]
        test_files = files[test_split_idx:]

        for file_name in train_files:
            shutil.copy(os.path.join(class_path, file_name), train_class_path)

        for file_name in val_files:
            shutil.copy(os.path.join(class_path, file_name), val_class_path)

        for file_name in test_files:
            shutil.copy(os.path.join(class_path, file_name), test_class_path)

        print(f"Class {class_name}: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} testing")

if __name__ == "__main__":
    input_directory = "data/normalized_landmarks"
    train_directory = "data/train_landmarks"
    val_directory = "data/val_landmarks"
    test_directory = "data/test_landmarks"

    split_dataset(input_directory, train_directory, val_directory, test_directory, val_ratio=0.1, test_ratio=0.2)
