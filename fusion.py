import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
import os

# Function to load models
def load_models(model_paths):
    models = []
    for path in model_paths:
        if os.path.exists(path):
            model = load_model(path)
            models.append(model)
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
    return models

# Function to preprocess input data for each model
def preprocess_input(model_name, x):
    """
    Preprocess input data based on the model type.
    """
    if model_name == 'cnn_model':
        # Resize to (64, 64) and normalize for the custom CNN model
        resized_images = tf.image.resize(x, (64, 64))
        normalized_images = resized_images / 255.0
    elif model_name == 'densenet_model':
        # Resize to (224, 224) and preprocess for DenseNet
        resized_images = tf.image.resize(x, (224, 224))
        normalized_images = densenet_preprocess_input(resized_images)
    elif model_name == 'resnet_model':
        # Resize to (224, 224) and preprocess for ResNet
        resized_images = tf.image.resize(x, (224, 224))
        normalized_images = resnet_preprocess_input(resized_images)
    else:
        raise ValueError(f"Unknown model name '{model_name}'")
    return normalized_images

# Function to average predictions from multiple models
def average_predictions(models, test_data, model_names, batch_size=32):
    """
    Compute the average predictions from multiple models.
    Supports batch-wise prediction to handle large datasets.
    """
    if len(models) != len(model_names):
        raise ValueError("The number of models and model names must match.")
    
    num_samples = test_data.shape[0]
    predictions = []

    # Predict in batches
    for i in range(0, num_samples, batch_size):
        batch_data = test_data[i:i+batch_size]
        
        batch_predictions = []
        for j, model in enumerate(models):
            processed_data = preprocess_input(model_names[j], batch_data)  # Preprocess for the current model
            pred = model.predict(processed_data, verbose=0)
            batch_predictions.append(pred)
        
        # Average predictions for the current batch
        avg_batch_pred = np.mean(batch_predictions, axis=0)
        predictions.append(avg_batch_pred)
    
    # Combine all batch predictions into one array
    avg_pred = np.concatenate(predictions, axis=0)
    return avg_pred

if __name__ == "__main__":
    # Paths to your saved models (.h5 files)
    model_paths = [
        'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\models\\cnn_model.h5',
        'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\models\\densenet_model.h5',
        'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\models\\resnet_model.h5'
    ]

    # Path to your test data (.npy file)
    test_data_path = 'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\data\\X_test.npy'

    # Load test data
    if os.path.exists(test_data_path):
        x_test = np.load(test_data_path)
        if len(x_test.shape) == 3:  # If missing batch dimension
            x_test = np.expand_dims(x_test, axis=0)
    else:
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    # Model names corresponding to each model in model_paths
    model_names = ['cnn_model', 'densenet_model', 'resnet_model']

    # Load models
    try:
        models = load_models(model_paths)
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)

    # Get average predictions
    try:
        avg_pred = average_predictions(models, x_test, model_names)
        print(f'Average predictions shape: {avg_pred.shape}')
    except Exception as e:
        print(f"Error during prediction: {e}")
