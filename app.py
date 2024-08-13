import os
import torch
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
from model import model
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import gridfs
import math

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/Emails"
mongo = PyMongo(app)
fs = gridfs.GridFS(mongo.db)

app.secret_key = 'iubiuh467t4876A'

# Load the trained model
model_path = "siamesemodel6.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def calculate_threshold(model, original_image_path, test_image_path, transform, device):
    original_image = Image.open(original_image_path).convert("L")
    test_image = Image.open(test_image_path).convert("L")

    if transform is not None:
        original_image = transform(original_image)
        test_image = transform(test_image)

    original_image = original_image.unsqueeze(0).to(device)
    test_image = test_image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        features1 = model(original_image)
        features2 = model(test_image)
    
    euclidean_distance = torch.nn.functional.pairwise_distance(features1, features2).mean().item()
    threshold = euclidean_distance + 0.01
    
    return threshold

def predict_signature(model, original_image_path, test_image_path, transform, device, threshold):
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
    if(euclidean_distance<threshold):
        return "Original"
    else:
        return "Forged"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/login_page')
def login_page():
    return render_template('login.html')

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 403

    user_id = session['user_id']
    
    # Fetch user details from MongoDB
    user = mongo.db.users.find_one({'email': user_id})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    threshold = user.get('threshold')
    if threshold is None:
        return jsonify({'error': 'User threshold not found'}), 404
    
    try:
        file1 = fs.get(user['file1_id']).read()
        file2 = fs.get(user['file2_id']).read()
        
        with open('temp1.png', 'wb') as f:
            f.write(file1)
        with open('temp2.png', 'wb') as f:
            f.write(file2)
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve files: {str(e)}'}), 500
    
    '''file1 = request.files['file1']
    file2 = request.files['file2']'''
    
    file1.save('temp1.png')
    file2.save('temp2.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_genuine = predict_signature(model, 'temp1.png', 'temp2.png', transform, device, threshold)

    os.remove('temp1.png')
    os.remove('temp2.png')

    return jsonify({'is_genuine': is_genuine})

@app.route('/register', methods=['POST'])
def register():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Please provide two images'}), 400

    email = request.form.get('email')
    username = request.form.get('username')
    password = request.form.get('password')
    file1 = request.files['file1']
    file2 = request.files['file2']

    file1_id = fs.put(file1, filename=f"{email}1.png")
    file2_id = fs.put(file2, filename=f"{email}2.png")

    file1_path = 'temp1.png'
    file2_path = 'temp2.png'
    
    with open(file1_path, 'wb') as f:
        f.write(fs.get(file1_id).read())
    with open(file2_path, 'wb') as f:
        f.write(fs.get(file2_id).read())
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    user = mongo.db.users.find_one({'email': email})
    if user:
        return jsonify({'message': 'User already exists'}), 400

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold = calculate_threshold(model, file1_path, file2_path, transform, device)

    mongo.db.users.insert_one({
        'username':username,
        'email': email,
        'password': hashed_password,
        'file1_id': file1_id,
        'file2_id': file2_id,
        'threshold': threshold
    })

    os.remove(file1_path)
    os.remove(file2_path)

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = mongo.db.users.find_one({'email': email})
    if not user or not bcrypt.check_password_hash(user['password'], password):
        return jsonify({'message': 'Invalid credentials'}), 401
    
    session['user_id'] = email

    return jsonify({'message': 'Login successful'}), 200

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None) 
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/compare', methods=['POST'])
def check():
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 403

    user_id = session['user_id']
    
    # Fetch user details from MongoDB
    user = mongo.db.users.find_one({'email': user_id})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    threshold = user.get('threshold')
    if threshold is None:
        return jsonify({'error': 'User threshold not found'}), 404

    # Retrieve and save the stored images
    try:
        file1 = fs.get(user['file1_id']).read()
        file2 = fs.get(user['file2_id']).read()
        
        with open('temp1.png', 'wb') as f:
            f.write(file1)
        with open('temp2.png', 'wb') as f:
            f.write(file2)
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve files: {str(e)}'}), 500
    
    # Save the uploaded image
    if 'file0' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file0 = request.files['file0']
    file0.save('temp3.png')

    # Perform the comparison
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        d1 = calculate_threshold(model, 'temp1.png', 'temp3.png', transform, device)
        d2 = calculate_threshold(model, 'temp2.png', 'temp3.png', transform, device)
        print(d1)
        print(d2)
        if (d1<=threshold or d2<=threshold):
            is_genuine = 'Original'
        else:
            is_genuine= 'Forged'
    except Exception as e:
        return jsonify({'error': f'Failed to perform comparison: {str(e)}'}), 500
    finally:
        # Clean up temporary files
        os.remove('temp1.png')
        os.remove('temp2.png')
        os.remove('temp3.png')

    return jsonify({'is_genuine': is_genuine})


@app.route('/check-login-status', methods=['GET'])
def check_login_status():
    if 'user_id' in session:
        return jsonify({'loggedIn': True})
    else:
        return jsonify({'loggedIn': False})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)

