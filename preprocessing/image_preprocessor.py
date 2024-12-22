from PIL import Image
import os

def preprocess_images(input_dir, output_dir):
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if not os.path.isdir(label_path):
            continue

        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)

        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = Image.open(image_path).convert('L').resize((28, 28))
            image.save(os.path.join(output_label_dir, image_name))

preprocess_images("data/raw", "data/train")
