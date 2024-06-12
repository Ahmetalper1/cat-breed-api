from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Update the model path if necessary
MODEL_PATH = 'C:/Users/HP/OneDrive/Masaüstü/Cat Breeds Classification/Data/models/20240608-150235/model-cat-vision-imagenet.keras'

# Load the model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


# Define the cat breeds
breeds = [
    'Abyssinian', 'American Bobtail', 'American Shorthair', 'Bengal', 'Birman',
    'Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Persian',
    'Ragdoll', 'Russian Blue', 'Siamese', 'Sphynx', 'Tuxedo'
]

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    breed = breeds[predicted_class[0]]
    
    return breed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        breed = predict_image(file_path)
        os.remove(file_path)
        return jsonify({'breed': breed})

if __name__ == '__main__':
    app.run(debug=True)
