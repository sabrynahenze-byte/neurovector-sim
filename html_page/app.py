#!/usr/bin/env python3

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import io
import base64

# Set matplotlib to use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Define the EXACT same architecture used for training
class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Load model globally
MODEL_PATH = "./my_trained_mnist.pt"  # Update this path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = SimpleMNIST().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

def preprocess_image(image_bytes):
    """Preprocess uploaded image for MNIST model"""
    
    # Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Apply transforms and add batch dimension
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img

def create_result_visualization(image, probabilities, prediction):
    """Create a visualization of the prediction without GUI"""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the image
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Uploaded Image\nPrediction: {prediction}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Plot probabilities
    digits = list(range(10))
    colors = ['green' if d == prediction else 'gray' for d in digits]
    bars = ax2.bar(digits, probabilities, color=colors)
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability')
    ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    
    # Add value labels
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        if height > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save to bytes buffer instead of file
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Convert to base64
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Clean up
    plt.close(fig)
    buf.close()
    
    return plot_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # CORS preflight
        return '', 200
    
    print("\n" + "="*50)
    print("PREDICTION REQUEST RECEIVED")
    print("="*50)
    
    if model is None:
        print("Model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        print("No file in request")
        print(f"Request files: {request.files}")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}")
    print(f"Content type: {file.content_type}")
    
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        print("Reading image bytes...")
        img_bytes = file.read()
        print(f"Image size: {len(img_bytes)} bytes")
        
        print("Preprocessing image...")
        img_tensor, original_img = preprocess_image(img_bytes)
        img_tensor = img_tensor.to(device)
        print(f"Image preprocessed, tensor shape: {img_tensor.shape}")
        
        # Make prediction
        print("Making prediction...")
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.exp(output)
            prediction = probs.argmax(dim=1).item()
            confidence = probs.max().item()
            all_probs = probs.squeeze().tolist()
        
        print(f"Prediction: {prediction} with confidence: {confidence:.4f}")
        
        # Get top 3 predictions
        probs_with_digits = list(enumerate(all_probs))
        probs_with_digits.sort(key=lambda x: x[1], reverse=True)
        top3 = [{'digit': d, 'probability': f'{p:.2%}'} for d, p in probs_with_digits[:3]]
        print(f"Top 3: {top3}")
        
        # Create visualization
        print("Creating visualization...")
        plot_base64 = create_result_visualization(original_img, all_probs, prediction)
        print("Visualization created")
        
        # Prepare response
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': f'{confidence:.2%}',
            'top3': top3,
            'plot': plot_base64,
            'probabilities': {str(i): f'{p:.4f}' for i, p in enumerate(all_probs)}
        }
        
        print("Sending response back to client")
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
