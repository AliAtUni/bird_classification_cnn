from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from src.model import build_model
import json

# Load the label mapping
with open('int_class_id_to_label.json', 'r') as f:
    label_map = json.load(f)


app = Flask(__name__)

# Load your trained model (adjust according to your actual setup)
model = build_model(num_classes=525)  # Adjust if necessary
model.load_state_dict(torch.load('models/bird_classification_model.pth', map_location=torch.device('cpu')))  # Map to CPU
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes))
            img = transform(img)
            img = img.unsqueeze(0)  # Add batch dimension
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            prediction_index = predicted.item()  # Get predicted index
            prediction_label = label_map.get(str(prediction_index), "Unknown label")  # Lookup readable label
            return jsonify(result=prediction_label)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
