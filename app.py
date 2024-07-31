import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import model
import torchvision.transforms as transforms
from compare import predict_signature
from extract import box_extraction
from fine_dataset import FineTuneDataset
from torch.utils.data import DataLoader
from fine_tune import fine_tune

# Initialize the Flask app
app = Flask(__name__)
CORS(app)


model_path = "siamesemodel2 (1).pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

    signatures = box_extraction('temp1.png')

    fine_dataset = FineTuneDataset(signatures = signatures, transform = transform)

    fine_dataloader = DataLoader(fine_dataset, shuffle = True, batch_size = 10, num_workers = 0)

    model_tuned = fine_tune(fine_dataloader= fine_dataloader, model = model, device=device)
    
    is_genuine = predict_signature(model_tuned, 'temp1.png', 'temp2.png', transform, device)

    os.remove('temp1.png')
    os.remove('temp2.png')

    return jsonify({'is_genuine': is_genuine})
