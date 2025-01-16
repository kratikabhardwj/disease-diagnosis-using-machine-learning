import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Import accuracy_score
import matplotlib.pyplot as plt
import cv2  # Import cv2 here for image processing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load test data
data_dir = 'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\data'
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

# Ensure X_test is resized correctly
DEFAULT_IMG_SIZE = (224, 224)

def preprocess_image(image):
    image = cv2.resize(image, DEFAULT_IMG_SIZE)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Preprocess the test data
X_test = np.array([preprocess_image(img) for img in X_test])

# Load the trained model
model_path = 'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\models\\best_cnn_model.h5'
cnn_model = load_model(model_path)

# Evaluate the model
y_pred = cnn_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Print model accuracy
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report and confusion matrix
logging.info("Classification Report:")
logging.info(classification_report(y_test, y_pred))

logging.info("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
logging.info(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Pneumonia'])
plt.yticks(tick_marks, ['Normal', 'Pneumonia'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
