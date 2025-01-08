from flask import Flask, request, jsonify
import torch.nn.functional as F 
import torch
from train_model import SignLanguageModel
import mediapipe as mp
import cv2
import numpy as np
import base64
import os
import sys
from preprocess_landmarks import normalize_landmarks_by_bounding_box
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

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

        normalized_landmarks = normalize_landmarks_by_bounding_box(raw_landmarks)

        flattened_landmarks = [value for lm in normalized_landmarks for value in lm.values()]

        max_landmark_size = 126
        if len(flattened_landmarks) < max_landmark_size:
            flattened_landmarks.extend([0.0] * (max_landmark_size - len(flattened_landmarks)))
        elif len(flattened_landmarks) > max_landmark_size:
            flattened_landmarks = flattened_landmarks[:max_landmark_size]

        input_tensor = torch.FloatTensor(flattened_landmarks).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1) 
            confidence, prediction = torch.max(probabilities, 1)
        
        confidence_score = confidence.item()  

        predicted_sign = idx_to_class[prediction.item()]

        return jsonify({"sign": predicted_sign, "confidence": confidence_score}) 
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
