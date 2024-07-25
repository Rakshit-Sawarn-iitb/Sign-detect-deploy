import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
from model import model

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = "siamesemodel2 (1).pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_signature(model, original_image_path, test_image_path, transform, device, threshold=0.001):
    original_image = Image.open(original_image_path).convert("L")
    test_image = Image.open(test_image_path).convert("L")

    if transform is not None:
        original_image = transform(original_image)
        test_image = transform(test_image)

    original_image = original_image.unsqueeze(0).to(device)
    test_image = test_image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        features1 = model.forward_features(original_image)
        features2 = model.forward_features(test_image)
    
    euclidean_distance = torch.nn.functional.pairwise_distance(features1, features2).mean().item()
    print(euclidean_distance)

    if euclidean_distance < threshold:
        response = "Original"
    else:
        response = "Forged"
    
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Please provide two images'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    file1.save('temp1.png')
    file2.save('temp2.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_genuine = predict_signature(model, 'temp1.png', 'temp2.png', transform, device)

    os.remove('temp1.png')
    os.remove('temp2.png')

    return jsonify({'is_genuine': is_genuine})
