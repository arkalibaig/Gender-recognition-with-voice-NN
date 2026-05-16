# Voice Gender Recognition with Neural Networks

A professional-grade Artificial Neural Network (ANN) for classifying gender based on acoustic voice features.

## Project Overview
This project implements a deep learning approach to identify gender from processed vocal data. It uses a multi-layer perceptron (MLP) architecture built with TensorFlow/Keras, achieving near-perfect accuracy on standardized acoustic datasets.

### Recent Fixes
*   **Feature Extraction Alignment:** Resolved a critical mismatch between training and inference feature extraction. Specifically, corrected the spectral contrast calculation and ensured the feature vector exactly matches the 43-dimensional input expected by the model, preventing incorrect high-confidence predictions on real audio.
*   **Regularization:** Enhanced model robustness against overfitting by introducing L2 regularization and dropout layers in the training pipeline.

## Acoustic Features
The model utilizes 43 acoustic features extracted from raw audio:
*   **Spectral Features:** Centroid (mean, std), Bandwidth (mean, std), Contrast (mean), Flatness (mean), Rolloff (mean).
*   **Time Domain Features:** Zero Crossing Rate (mean), RMS Energy (mean).
*   **Pitch Features:** Mean, Min, Max, and Std of the fundamental frequency (extracted via Yin algorithm).
*   **Statistical Measures:** Spectral skew and kurtosis.
*   **Energy Measures:** Energy entropy and log energy.
*   **MFCCs:** Mean and standard deviation for the first 13 Mel-Frequency Cepstral Coefficients.


## Directory Structure
*   `data/`: Contains the processed acoustic feature dataset.
*   `models/`: Saved model architectures and trained weights.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA), model research, and evaluation.
*   `scripts/`: Python scripts for data preprocessing, automated training, and evaluation.
*   `UI/`: Modern HTML/CSS frontend for the live demo.
*   `app.py`: FastAPI backend to serve the model for real-time inference.
*   `requirements.txt`: Project dependencies.

## Technical Stack
*   **Deep Learning:** TensorFlow, Keras
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Feature Extraction:** Librosa (used for extracting acoustic features)
*   **Model Management:** Joblib (for saving/loading scalers), H5 (for Keras models)
*   **Backend:** FastAPI, Uvicorn
*   **Frontend:** HTML5, CSS3, JavaScript

## How to Run

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required libraries:
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### 2. Preprocess Data and Train the Model
The `scripts/train.py` script handles data loading, feature extraction, scaling, model training, and saving.
```bash
python scripts/train.py
```
This process may take several minutes depending on your hardware.

### 3. Model Inference (Prediction)
To predict the gender of a new audio file, use the `scripts/predict.py` script.
```bash
python scripts/predict.py --audio_file path/to/your/audio.wav
```
The script will output the predicted gender and confidence score.

### 4. Live Demo (Web Interface)
You can run a local server to use the interactive web UI:
1. Start the backend server:
   ```bash
   python app.py
   ```
2. Open `UI/index.html` in your web browser.
3. Upload an audio file to see the prediction in real-time.

### 5. Research and Development
Explore the notebooks in `notebooks/` for detailed analysis, model architecture insights, and evaluation metrics.

## Model Performance
The current MLP architecture consistently achieves **99%+ accuracy** on the test set. It incorporates early stopping and dropout layers to mitigate overfitting and enhance generalization.

## Contribution Guidelines
We welcome contributions! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch for your feature or fix.
3.  Make your changes and ensure all tests pass.
4.  Submit a pull request.

## Future Work
*   Explore alternative model architectures (e.g., Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN)) for potentially improved performance.
*   Enhance robustness against background noise in audio samples.

## License
This project is licensed under the MIT License.
