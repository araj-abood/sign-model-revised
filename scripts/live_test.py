import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.helpers import normalize_landmarks  
from scripts.train_model import SignLanguageModel 

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def live_test(model_path, idx_to_class):
    """
    Perform live testing using the webcam and display predictions.

    Args:
        model_path (str): Path to the trained model file.
        idx_to_class (dict): Mapping from class indices to class labels.
    """
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageModel(num_classes=len(idx_to_class))
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state"])
    model.to(device)
    model.eval()

    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # Start the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Retrying...")
            continue

        # Flip the frame for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Display landmarks on the frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks and normalize
                landmarks = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in hand_landmarks.landmark
                ]
                normalized_landmarks = normalize_landmarks(landmarks)

                # Convert to a PyTorch tensor
                input_tensor = torch.tensor([normalized_landmarks], dtype=torch.float32).to(device)

                # Perform prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted_idx = torch.max(outputs, 1)
                    predicted_label = idx_to_class[predicted_idx.item()]

                    # Display prediction on the frame
                    cv2.putText(frame, f"Prediction: {predicted_label}",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Live Test", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "data/models/sign_language_model.pth"

    idx_to_class = {
        0: "السلام عليكم",
        1: "شكرا",
        2: "مرحباً",
    }

    live_test(model_path, idx_to_class)
