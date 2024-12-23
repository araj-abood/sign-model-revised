import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import os



def decode_image(base64_string):
    """Decode a base64-encoded image string."""
    if 'data:image' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def render_arabic_text(frame, text, position, font_path="d:/programming/grad-project/sign-ai-model/fonts/Amiri-Regular.ttf", font_size=32, color=(0, 255, 0)):

    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    
    font = ImageFont.truetype(font_path, font_size)
   

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Render the text
    draw.text(position, bidi_text, font=font, fill=(color[2], color[1], color[0]))

    # Convert back to OpenCV frame
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


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


