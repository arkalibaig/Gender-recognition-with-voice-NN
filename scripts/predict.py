import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
import argparse
from scipy.stats import skew, kurtosis

def extract_features(audio_path):
    """
    Extracts the 44 acoustic features required by the model.
    Matches the columns in vocal_gender_features.csv.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Spectral Features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # Time Domain Features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    rms_energy = librosa.feature.rms(y=y)[0]
    
    # Pitch Features (using Yin algorithm)
    pitches = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitches = pitches[~np.isnan(pitches)]
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Aggregate Features
    features = [
        np.mean(spectral_centroids),
        np.std(spectral_centroids),
        np.mean(spectral_bandwidths),
        np.std(spectral_bandwidths),
        np.mean(spectral_contrast),
        np.mean(spectral_flatness),
        np.mean(spectral_rolloff),
        np.mean(zero_crossing_rate),
        np.mean(rms_energy),
        np.mean(pitches) if len(pitches) > 0 else 0,
        np.min(pitches) if len(pitches) > 0 else 0,
        np.max(pitches) if len(pitches) > 0 else 0,
        np.std(pitches) if len(pitches) > 0 else 0,
        skew(spectral_centroids),
        kurtosis(spectral_centroids),
        # Energy Entropy (simplified)
        -np.sum(rms_energy**2 * np.log(rms_energy**2 + 1e-10)), 
        np.log(np.sum(y**2) + 1e-10)
    ]
    
    # Add MFCC means and stds
    for i in range(13):
        features.append(np.mean(mfccs[i]))
        features.append(np.std(mfccs[i]))
        
    return np.array(features).reshape(1, -1)

def predict(audio_file):
    """Main prediction function."""
    model_path = "models/gender_recognition_model.h5"
    scaler_path = "models/scaler.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Model or scaler not found. Please run training first.")
        return

    # Load artifacts
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Analyzing audio: {audio_file}...")
    try:
        features = extract_features(audio_file)
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled, verbose=0)[0][0]
        gender = "Female" if prediction > 0.5 else "Male"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        print(f"\nResult: {gender}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"Error during feature extraction or prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gender recognition from voice audio.")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to the audio file.")
    args = parser.parse_args()
    
    predict(args.audio_file)
