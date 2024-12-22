import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display




def decode_image(base64_string):
    """Decode a base64-encoded image string."""
    if 'data:image' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def render_arabic_text(frame, text, position, font_path="../fonts/Amiri-Regular.ttf", font_size=32, color=(0, 255, 0)):
    """
    Render Arabic text on an OpenCV frame using Pillow.
    
    Args:
        frame (np.ndarray): The OpenCV frame.
        text (str): The Arabic text to render.
        position (tuple): (x, y) position to render the text.
        font_path (str): Path to the TTF font file supporting Arabic.
        font_size (int): Font size.
        color (tuple): Text color (B, G, R).
    """
    # Reshape and bidi process Arabic text
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    # Convert OpenCV frame to Pillow Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Render the text
    draw.text(position, bidi_text, font=font, fill=(color[2], color[1], color[0]))

    # Convert back to OpenCV frame
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
