from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import json
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# Load model
model = load_model("dish_model.h5", custom_objects={"preprocess_input": preprocess_input})

# Load classes
try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
except Exception as e:
    logging.warning(f"⚠️ Failed to load class_names.json: {e}")
    class_names = []

# Load ingredients
try:
    with open("ingredients.json", "r") as f:
        ingredient_data = json.load(f)
except Exception as e:
    logging.warning(f"⚠️ Failed to load ingredients.json: {e}")
    ingredient_data = {}

# Validate class_names vs model output
try:
    num_classes = int(model.output_shape[-1])
    if len(class_names) != num_classes:
        logging.error(f"Class count mismatch: class_names={len(class_names)} vs model={num_classes}")
except Exception as e:
    logging.warning(f"Could not infer model output shape: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image: Image.Image):
    # Resize and ensure RGB
    image = image.resize((224, 224)).convert('RGB')
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    # IMPORTANT: use EfficientNet preprocessing (matches training)
    arr = preprocess_input(arr)
    return arr

@app.before_request
def log_request_info():
    logging.info(f"Incoming request: {request.method} {request.path}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        image = Image.open(image_file)
        input_tensor = preprocess_image(image)

        # Get predictions
        prediction = model.predict(input_tensor)[0]

        # Basic sanity: if class_names mismatch, return a clear error
        if len(class_names) != len(prediction):
            logging.error(f"Mismatch: len(class_names)={len(class_names)} len(prediction)={len(prediction)}")
            return jsonify({'error': 'Server label configuration mismatch. Please refresh class_names.json.'}), 500

        # Top-3
        top_indices = prediction.argsort()[-3:][::-1]
        logging.info(f"Top indices: {top_indices.tolist()}")
        logging.info(f"Dishes: {[class_names[i] for i in top_indices]}")
        logging.info(f"Confidences: {[round(float(prediction[i])*100,2) for i in top_indices]}")

        top_results = []
        for i in top_indices:
            dish_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            confidence = round(float(prediction[i]) * 100, 2)

            # Robust ingredient lookup: exact, lowercase, underscore-normalized
            ingredients = (
                ingredient_data.get(dish_name)
                or ingredient_data.get(dish_name.lower())
                or ingredient_data.get(dish_name.replace(" ", "_"))
                or ingredient_data.get(dish_name.replace(" ", "_").lower())
                or ["Ingredients not available. Please update ingredients.json."]
            )

            top_results.append({
                "dish": dish_name,
                "confidence": confidence,
                "ingredients": ingredients
            })

        return jsonify({"predictions": top_results})

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/version', methods=['GET'])
def version():
    return jsonify({
        "model": "dish_model.h5",
        "classes": len(class_names),
        "status": "running"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
