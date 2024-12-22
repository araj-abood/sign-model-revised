import torch
from torchvision import transforms
from PIL import Image
from detection_model import SignLanguageModel

# Load the model
num_classes = 26  # Adjust based on your dataset
model = SignLanguageModel(num_classes=num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Prediction
def predict(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Test
label_map = {0: "A", 1: "B", 2: "C"}  # Update based on your dataset
image_path = "data/test/A/test_image.jpg"
prediction = predict(image_path)
print(f"Predicted Sign: {label_map[prediction]}")
