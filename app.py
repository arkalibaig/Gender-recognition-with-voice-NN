import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import skew, kurtosis
import shutil
import tempfile

app = FastAPI()

# Enable CORS for the UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = "/home/arkalibaig/gender_reco_nn/models/gender_recognition_model.h5"
SCALER_PATH = "/home/arkalibaig/gender_reco_nn/models/scaler.joblib"

# Load model and scaler once
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Model or scaler not found. Please check the paths.")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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
    ]
    
    # Add all 7 bands of spectral contrast
    # spectral_contrast is (7, frames), we take mean across frames for each band
    for band in range(7):
        features.append(np.mean(spectral_contrast[band]))
        
    features.extend([
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
    ])

    
    # Add MFCC means and stds
    for i in range(13):
        features.append(np.mean(mfccs[i]))
        features.append(np.std(mfccs[i]))
        
    return np.array(features).reshape(1, -1)

@app.post("/predict")
async def predict_gender(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Extract features
        features = extract_features(tmp_path)
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction_prob = model.predict(features_scaled, verbose=0)[0][0]
        
        # Format response
        gender = "Female" if prediction_prob > 0.5 else "Male"
        confidence = float(prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob)
        
        # Clean up
        os.remove(tmp_path)
        
        return {
            "prediction": gender,
            "confidence": confidence
        }
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
