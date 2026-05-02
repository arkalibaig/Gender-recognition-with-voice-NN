# Voice Gender Recognition with Neural Networks

A professional-grade Artificial Neural Network (ANN) for classifying gender based on acoustic voice features.

## Project Overview
This project implements a deep learning approach to identify gender from processed vocal data. It uses a multi-layer perceptron (MLP) architecture built with TensorFlow/Keras, achieving near-perfect accuracy on standardized acoustic datasets.

## Directory Structure
*   `data/`: Contains the acoustic feature dataset.
*   `models/`: Saved model architectures and trained weights.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and research.
*   `scripts/`: Python scripts for automated training and evaluation.
*   `requirements.txt`: Project dependencies.

## Technical Stack
*   **Deep Learning:** TensorFlow, Keras
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Feature Scaling:** StandardScaler
*   **Model Management:** Joblib (for scalers), H5 (for models)

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the automated training script to preprocess data, train the ANN, and save the resulting model:
```bash
python scripts/train.py
```

### 3. Research and Development
Explore the research notebook in `notebooks/research.ipynb` for detailed analysis and model evaluation metrics.

## Model Performance
The current architecture consistently achieves **99%+ accuracy** on the test set, utilizing early stopping and dropout layers to prevent overfitting.

## License
MIT
