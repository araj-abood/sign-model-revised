import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import torch
import numpy as np
from utils.helpers import normalize_landmarks
from PIL import ImageFont
from preprocess_landmarks import normalize_landmarks_by_bounding_box
from utils.helpers import render_arabic_text
import mediapipe as mp

MODEL_PATH = "d:/programming/grad-project/sign-ai-model/data/model/sign_language_model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

from train_model import SignLanguageModel 

input_size = checkpoint.get('input_size', 128)  
num_classes = checkpoint.get('num_classes', 26) 

if input_size is None or num_classes is None:
    raise ValueError("Checkpoint is missing required keys 'input_size' or 'num_classes'.")

model = SignLanguageModel(input_size=input_size, num_classes=num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_to_idx = checkpoint.get('class_to_idx', {})
idx_to_class = {v: k for k, v in class_to_idx.items()}

print("Resolved MODEL_PATH:", os.path.abspath(MODEL_PATH))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


def predict_landmarks(landmarks):

    input_tensor = torch.tensor([landmarks], dtype=torch.float32)
    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_label_idx = torch.argmax(predictions, dim=1).item()
    return idx_to_class.get(predicted_label_idx, "Unknown")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to exit the application.")

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1) 
    if not ret:
        print("Failed to grab frame.")
        break

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    detected_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            detected_landmarks.append([
                {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark
            ])

    if detected_landmarks:
        print("Raw Detected Landmarks:", detected_landmarks)
        normalized_landmarks = normalize_landmarks_by_bounding_box(detected_landmarks[0])
        print("Normalized Landmarks:", normalized_landmarks)
        flattened_landmarks = [value for lm in normalized_landmarks for value in lm.values()]
        print("Flattened Landmarks:", flattened_landmarks)

        if len(flattened_landmarks) < input_size:
            flattened_landmarks.extend([0.0] * (input_size - len(flattened_landmarks)))
        elif len(flattened_landmarks) > input_size:
            flattened_landmarks = flattened_landmarks[:input_size]

        print("Input Tensor Shape:", len(flattened_landmarks))

        prediction = predict_landmarks(flattened_landmarks)
        print("Predicted Class:", prediction)

        label_text = f"Prediction: {prediction}"
        frame = render_arabic_text(frame, label_text, position=(50, 50))

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
