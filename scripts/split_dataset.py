import os
import shutil
import random

def split_dataset(input_dir, train_dir, test_dir, test_ratio=0.2):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        train_class_path = os.path.join(train_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        files = os.listdir(class_path)
        random.shuffle(files)

        split_idx = int(len(files) * (1 - test_ratio))
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for file_name in train_files:
            shutil.copy(os.path.join(class_path, file_name), train_class_path)

        for file_name in test_files:
            shutil.copy(os.path.join(class_path, file_name), test_class_path)

        print(f"Class {class_name}: {len(train_files)} training, {len(test_files)} testing")

if __name__ == "__main__":
    input_directory = "data/normalized_landmarks"
    train_directory = "data/train_landmarks"
    test_directory = "data/test_landmarks"

    split_dataset(input_directory, train_directory, test_directory, test_ratio=0.2)
