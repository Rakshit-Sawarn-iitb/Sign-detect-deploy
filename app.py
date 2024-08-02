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
from pymongo import MongoClient


# Initialize the Flask app
app = Flask(__name__)
CORS(app)


mongodb_uri = 'mongodb://localhost:27017/'
client = MongoClient(mongodb_uri)
db = client['Emails']
collection = db['user_emails']

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

    print("Checking Started")

    signatures = box_extraction('temp1.png')

    print("Boxes made")

    fine_dataset = FineTuneDataset(signatures = signatures, transform = transform)

    print("Dataset Created")

    print(len(fine_dataset))

    fine_dataloader = DataLoader(fine_dataset, shuffle = True, batch_size = 10, num_workers = 0)

    print("Dataloader created")

    model_tuned = fine_tune(fine_dataloader= fine_dataloader, model = model, device=device)

    print("Model Tuned")
    
    is_genuine = predict_signature(model_tuned, signatures, 'temp2.png', transform, device)

    print("Prediction done")

    os.remove('temp1.png')
    os.remove('temp2.png')

    return jsonify({'is_genuine': is_genuine})

@app.route('/submit_email', methods=['POST'])
def submit_email():
    email = request.json.get('email')
    if email:
        # Insert the email into the MongoDB collection
        collection.insert_one({'email': email})
        return jsonify({'message': 'Email saved successfully!'}), 200
    return jsonify({'error': 'No email provided'}), 400