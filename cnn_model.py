import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Function to create CNN model
def create_cnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),A
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to load data (replace with actual data loading later)
def load_data():
    x_train = np.random.rand(100, 64, 64, 3)  # Dummy data
    y_train = np.random.randint(2, size=100)
    return x_train, y_train

# Ensure the directory for saving the model exists
save_dir = 'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\models'
os.makedirs(save_dir, exist_ok=True)

# Function to train and save the model
def train_and_save_model():
    x_train, y_train = load_data()
    input_shape = x_train.shape[1:]
    model = create_cnn_model(input_shape)

    # Create ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        os.path.join(save_dir, 'best_cnn_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, callbacks=[checkpoint_callback])
    
    # Save the model after training
    model.save(os.path.join(save_dir, 'cnn_model.h5'))
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
