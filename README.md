# Voice Gender Recognition with Neural Networks

A professional-grade Artificial Neural Network (ANN) for classifying gender based on acoustic voice features.

## Project Overview
This project implements a deep learning approach to identify gender from processed vocal data. It uses a multi-layer perceptron (MLP) architecture built with TensorFlow/Keras, achieving near-perfect accuracy on standardized acoustic datasets.

## Acoustic Features
The model utilizes common acoustic features extracted from raw audio. These include:
*   **Mel-Frequency Cepstral Coefficients (MFCCs):** Represent the short-term power spectrum of a sound.
*   **Chroma Features:** Relate to the 12 different pitch classes.
*   **Mel Spectrogram:** A representation of the audio spectrum.
*   **Spectral Contrast:** Measures the difference between peaks and valleys in the spectrum.

These features capture various nuances of voice timbre, pitch, and rhythm that are indicative of gender.

## Directory Structure
*   `data/`: Contains the processed acoustic feature dataset.
*   `models/`: Saved model architectures and trained weights.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA), model research, and evaluation.
*   `scripts/`: Python scripts for data preprocessing, automated training, and evaluation.
*   `requirements.txt`: Project dependencies.

## Technical Stack
*   **Deep Learning:** TensorFlow, Keras
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Feature Extraction:** Librosa (used for extracting acoustic features)
*   **Model Management:** Joblib (for saving/loading scalers), H5 (for Keras models)

## How to Run

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required libraries:
```bash
pip install -r requirements.txt
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

### 4. Research and Development
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
*   Explore alternative model architectures (e.g., Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs)) for potentially improved performance.
*   Enhance robustness against background noise in audio samples.
*   Develop a simple API or web service for real-time gender prediction.

## License
This project is licensed under the MIT License.
