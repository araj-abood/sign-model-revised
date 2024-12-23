
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import mediapipe as mp
import json
import time
from bidi.algorithm import get_display
import numpy as np
from utils.helpers import render_arabic_text

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def capture_landmarks(output_dir, sign_labels, num_samples_per_sign):
    """
    Capture hand landmarks and save them in a structured format with a delay.

    Parameters:
        output_dir (str): Directory to save landmarks.
        sign_labels (list): List of sign labels (e.g., ['السلام عليكم', 'شكرا']).
        num_samples_per_sign (int): Number of samples to capture per sign.
    """
    os.makedirs(output_dir, exist_ok=True)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("\n=== Hand Landmark Collection ===")
    print("Instructions:")
    print("- Select a sign and press 'SPACE' to start capturing samples.")
    print("- Press 'q' to quit the collection process.\n")

    for sign_label in sign_labels:
        print(f"\nReady to collect landmarks for sign: {sign_label}")
        input("Press ENTER when ready to start...")

        sign_dir = os.path.join(output_dir, sign_label)
        os.makedirs(sign_dir, exist_ok=True)

        sample_count = 0
        capturing = False
        start_time = None

        while sample_count < num_samples_per_sign:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Retrying...")
                continue

            # Flip the frame for a mirrored view
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # Display landmarks on the frame for feedback
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the status and instructions
            frame = render_arabic_text(frame, f"Sign: {sign_label}", (10, 30), font_path="fonts/arial.ttf")
            cv2.putText(frame, f"Samples: {sample_count}/{num_samples_per_sign}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'SPACE' to capture | Press 'q' to quit",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Handle capturing logic
            if capturing:
                elapsed_time = time.time() - start_time
                if elapsed_time >= 2:  # 2-second delay
                    if result.multi_hand_landmarks:
                        # Combine landmarks from both hands
                        all_landmarks = []
                        for hand_landmarks in result.multi_hand_landmarks:
                            landmarks = [
                                {"x": lm.x, "y": lm.y, "z": lm.z}
                                for lm in hand_landmarks.landmark
                            ]
                            all_landmarks.append(landmarks)

                        # Save all combined landmarks to a JSON file
                        landmark_path = os.path.join(sign_dir, f"{sample_count}.json")
                        with open(landmark_path, "w") as f:
                            json.dump(all_landmarks, f, indent=4)

                        print(f"Saved landmarks: {landmark_path}")
                        sample_count += 1
                        capturing = False

            # Show the frame
            cv2.imshow("Hand Landmark Capture", frame)

            # Handle keyboard events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                print("Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):  # Start capturing
                capturing = True
                start_time = time.time()
                print("Capturing... Get ready!")

        print(f"Finished collecting landmarks for sign: {sign_label}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\nLandmark collection completed.")

if __name__ == "__main__":
    # Directory to save landmarks
    output_directory = "data/landmarks"

    # List of Arabic signs to capture
    signs = ["ثلاث", "اثنان", "واحد"]

    # Number of samples to capture per sign
    samples_per_sign = 20

    # Start capturing landmarks
    capture_landmarks(output_directory, signs, samples_per_sign)
