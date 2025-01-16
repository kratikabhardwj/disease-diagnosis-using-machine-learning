

import os
import cv2
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_IMG_SIZE = (224, 224)  
DATA_DIR = 'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\data'
SAVE_DIR = 'C:\\Users\\krati\\Desktop\\ml\\deep_fusion_disease_diagnosis\\data\\'


def preprocess_image(image_path, img_size=DEFAULT_IMG_SIZE):
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Could not read image: {image_path}")
            return None
        if len(image.shape) != 3 or image.shape[2] != 3:
            logging.warning(f"Invalid image dimensions: {image.shape} at {image_path}")
            return None
        image = cv2.resize(image, img_size)
        image = image / 255.0  
        return image
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None


def load_and_preprocess_data(data_dir=DATA_DIR, img_size=DEFAULT_IMG_SIZE):
    images, labels, failed_images = [], [], []
    subsets = ['train', 'test']
    
    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.exists(subset_dir):
            logging.error(f"Directory does not exist: {subset_dir}")
            continue

        for label in os.listdir(subset_dir):
            label_dir = os.path.join(subset_dir, label)
            if not os.path.isdir(label_dir):
                logging.warning(f"Skipping non-directory: {label_dir}")
                continue

            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                logging.info(f"Processing {image_path}")
                image = preprocess_image(image_path, img_size)
                if image is not None:
                    images.append(image)
                    labels.append(label)
                else:
                    failed_images.append(image_path)

    logging.info(f"Total failed images: {len(failed_images)}")
    return np.array(images), np.array(labels), failed_images


def save_label_mapping(label_mapping, save_dir=SAVE_DIR):
    mapping_path = os.path.join(save_dir, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f)
    logging.info(f"Label mapping saved to {mapping_path}")


def plot_class_distribution(labels):
    class_counts = Counter(labels)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.show()


def main():
    X, y, failed_images = load_and_preprocess_data(DATA_DIR)

    if len(X) == 0:
        logging.error("No images loaded. Exiting script.")
        exit()

    # Create label mapping
    label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
    y = np.array([label_mapping[label] for label in y])

    # Save data
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    save_label_mapping(label_mapping, SAVE_DIR)

    # Save the train, validation, and test data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(SAVE_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(SAVE_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(SAVE_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(SAVE_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(SAVE_DIR, 'y_test.npy'), y_test)

    logging.info(f"Data saved to {SAVE_DIR}")

    # Optionally plot class distribution
    plot_class_distribution(y)

if __name__ == '__main__':
    main()
