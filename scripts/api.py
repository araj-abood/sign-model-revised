from flask import Flask, request, jsonify
import torch
from train_model import SignLanguageModel
import mediapipe as mp
import cv2
import numpy as np
import base64
# Initialize Flask app
app = Flask(__name__)

# Load the model and metadata
MODEL_PATH = "data/model/sign_language_model.pth"
checkpoint = torch.load(MODEL_PATH)
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = SignLanguageModel(input_size=126, num_classes=len(class_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if 'landmarks' in data:
            raw_landmarks = data['landmarks']  # List of dictionaries
        else:
            # Decode and process the image to extract landmarks
            image_data = base64.b64decode(data['image'])
            np_img = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_image)

            if not result.multi_hand_landmarks:
                return jsonify({"error": "No hand landmarks detected"}), 400

            raw_landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for hand_landmarks in result.multi_hand_landmarks
                for lm in hand_landmarks.landmark
            ]

        # Normalize landmarks
        normalized_landmarks = normalize_landmarks(raw_landmarks)

        # Pad or truncate to model input size
        max_landmark_size = 126
        if len(normalized_landmarks) < max_landmark_size:
            normalized_landmarks.extend([0] * (max_landmark_size - len(normalized_landmarks)))
        elif len(normalized_landmarks) > max_landmark_size:
            normalized_landmarks = normalized_landmarks[:max_landmark_size]

        # Convert to tensor and predict
        input_tensor = torch.FloatTensor(normalized_landmarks).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, prediction = torch.max(output, 1)

        # Map prediction to class name
        predicted_sign = idx_to_class[prediction.item()]
        return jsonify({"sign": predicted_sign})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def normalize_landmarks(landmarks):
    """
    Normalize raw landmark data from dictionaries.

    Args:
        landmarks (list): List of dictionaries with keys 'x', 'y', 'z'.

    Returns:
        list: Normalized and flattened landmark values.
    """
    # Extract x, y, z values
    x_vals = [landmark['x'] for landmark in landmarks]
    y_vals = [landmark['y'] for landmark in landmarks]
    z_vals = [landmark['z'] for landmark in landmarks]

    # Find min/max for normalization
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    min_z, max_z = min(z_vals), max(z_vals)

    # Normalize coordinates to range [0, 1]
    normalized_landmarks = [
        {
            "x": (landmark['x'] - min_x) / (max_x - min_x) if max_x - min_x > 0 else 0,
            "y": (landmark['y'] - min_y) / (max_y - min_y) if max_y - min_y > 0 else 0,
            "z": (landmark['z'] - min_z) / (max_z - min_z) if max_z - min_z > 0 else 0,
        }
        for landmark in landmarks
    ]

    # Flatten the normalized coordinates into a single list
    return [value for coord in normalized_landmarks for value in coord.values()]

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=9000, debug=True)
