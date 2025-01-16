# Deep Fusion-Based Disease Diagnosis

## Overview
This project aims to develop a deep learning model that can diagnose diseases from medical images using a fusion of multiple deep learning models.

## Setup
1. Clone the repository.
2. Create a virtual environment:
    ```bash
    python -m venv env
    .\env\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project
1. Preprocess the data:
    ```bash
    python scripts/preprocess.py
    ```
2. Train the models:
    ```bash
    python scripts/train.py
    ```
3. Perform model fusion:
    ```bash
    python scripts/fusion.py
    ```
4. Evaluate the model:
    ```bash
    python scripts/evaluate.py
    ```
5. Deploy the model:
    ```bash
    python scripts/app.py
    ```
6. Send a POST request with an image to `/predict` to get a diagnosis.

## Dependencies
- tensorflow
- opencv-python
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- flask


juypter- http://localhost:8888/tree?token=14552320392b401b963e6b774bdc6a8ccede93eae118bf8a
