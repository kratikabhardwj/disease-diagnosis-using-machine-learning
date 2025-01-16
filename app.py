import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Paths to models
MODEL_PATHS = {
    "cnn_model": "C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\models\\cnn_model.h5",
    "densenet_model": "C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\models\\densenet_model.h5",
    "resnet_model": "C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\models\\resnet_model.h5"
}

# Load models
MODELS = {}
for model_name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        MODELS[model_name] = load_model(path)
    else:
        raise FileNotFoundError(f"Model file not found: {path}")

# Function to preprocess input image
def preprocess_input(model_name, image):
    """
    Preprocess input data based on the model type.
    """
    if model_name == "cnn_model":
        # Resize to (64, 64) and normalize for the custom CNN model
        resized_image = tf.image.resize(image, (64, 64))
        normalized_image = resized_image / 255.0
    elif model_name == "densenet_model":
        # Resize to (224, 224) and preprocess for DenseNet
        resized_image = tf.image.resize(image, (224, 224))
        normalized_image = densenet_preprocess_input(resized_image)
    elif model_name == "resnet_model":
        # Resize to (224, 224) and preprocess for ResNet
        resized_image = tf.image.resize(image, (224, 224))
        normalized_image = resnet_preprocess_input(resized_image)
    else:
        raise ValueError(f"Unknown model name '{model_name}'")
    return normalized_image

# Root endpoint
@app.route("/", methods=["GET"])
def home():
    return "Deep Fusion-Based Disease Diagnosis API. Use the /predict endpoint to get predictions."

# Favicon handling
@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return "", 204  # Return an empty response with HTTP status 204 (No Content)

# Route to predict from a single image
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Load the image
    file = request.files["file"]
    try:
        image = Image.open(file).convert("RGB")  # Ensure it's in RGB format
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    # Get predictions from each model
    predictions = []
    for model_name, model in MODELS.items():
        processed_image = preprocess_input(model_name, image_array)
        pred = model.predict(processed_image, verbose=0)
        predictions.append(pred)

    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)

    # Respond with the result
    result = {
        "average_prediction": avg_prediction.tolist(),
        "model_predictions": {name: pred.tolist() for name, pred in zip(MODEL_PATHS.keys(), predictions)}
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
