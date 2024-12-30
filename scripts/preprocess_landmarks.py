import os
import json

def normalize_landmarks_by_bounding_box(landmarks):
  
    x_coords = [lm["x"] for lm in landmarks]
    y_coords = [lm["y"] for lm in landmarks]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.append({
            "x": (lm["x"] - x_min) / (x_max - x_min),
            "y": (lm["y"] - y_min) / (y_max - y_min),
            "z": lm["z"]  
        })
    return normalized_landmarks

def preprocess_dataset(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for sign_label in os.listdir(input_dir):
        sign_input_path = os.path.join(input_dir, sign_label)
        sign_output_path = os.path.join(output_dir, sign_label)

        os.makedirs(sign_output_path, exist_ok=True)

        for file_name in os.listdir(sign_input_path):
            file_path = os.path.join(sign_input_path, file_name)

            with open(file_path, "r") as f:
                raw_landmarks = json.load(f)

            normalized_landmarks = []
            for hand_landmarks in raw_landmarks:
                normalized_landmarks.append(normalize_landmarks_by_bounding_box(hand_landmarks))

            output_path = os.path.join(sign_output_path, file_name)
            with open(output_path, "w") as f:
                json.dump(normalized_landmarks, f, indent=4)

            print(f"Processed: {output_path}")

if __name__ == "__main__":
    input_directory = "data/landmarks"  
    output_directory = "data/normalized_landmarks"  

    preprocess_dataset(input_directory, output_directory)
