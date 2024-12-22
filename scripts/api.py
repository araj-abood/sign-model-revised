from flask import Flask, request, jsonify
import torch
from train_model import SignLanguageModel 

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = "data/model/sign_language_model.pth"

# Load the model and metadata
checkpoint = torch.load(MODEL_PATH)
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = SignLanguageModel(input_size=126, num_classes=len(class_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests. Accepts JSON with landmark data.
    """
    try:
        data = request.json
        if 'landmarks' not in data:
            return jsonify({"error": "Missing 'landmarks' in request"}), 400

        landmarks = data['landmarks']

        # Flatten and pad/truncate landmarks to match model input size
        max_landmark_size = 126
        flattened = [val for hand in landmarks for lm in hand for val in lm.values()]
        if len(flattened) < max_landmark_size:
            flattened.extend([0] * (max_landmark_size - len(flattened)))
        elif len(flattened) > max_landmark_size:
            flattened = flattened[:max_landmark_size]

        # Convert to tensor
        input_tensor = torch.FloatTensor(flattened).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, prediction = torch.max(output, 1)

        # Map prediction to class name
        predicted_sign = idx_to_class[prediction.item()]
        return jsonify({"sign": predicted_sign})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
