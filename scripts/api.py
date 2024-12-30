from flask import Flask, request, jsonify
import torch
from train_model import SignLanguageModel
import mediapipe as mp
import cv2
import numpy as np
import base64
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.helpers import normalize_landmarks

app = Flask(__name__)

MODEL_PATH = "data/model/sign_language_model.pth"
checkpoint = torch.load(MODEL_PATH)
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = SignLanguageModel(input_size=126, num_classes=len(class_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

mp_hands = mp.solutions.hands

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if 'landmarks' in data:
            raw_landmarks = data['landmarks']  
        else:
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

        normalized_landmarks = normalize_landmarks(raw_landmarks)

        max_landmark_size = 126
        if len(normalized_landmarks) < max_landmark_size:
            normalized_landmarks.extend([0] * (max_landmark_size - len(normalized_landmarks)))
        elif len(normalized_landmarks) > max_landmark_size:
            normalized_landmarks = normalized_landmarks[:max_landmark_size]

        input_tensor = torch.FloatTensor(normalized_landmarks).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, prediction = torch.max(output, 1)

        predicted_sign = idx_to_class[prediction.item()]
        return jsonify({"sign": predicted_sign})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
