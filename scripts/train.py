import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(filepath):
    """Loads dataset and performs scaling."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

def build_model(input_shape):
    """Builds the Sequential ANN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train():
    """Main training pipeline."""
    data_path = "data/vocal_gender_features.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    (X_train, X_test, y_train, y_test), scaler = load_and_preprocess_data(data_path)
    
    model = build_model(X_train.shape[1])
    
    # Add early stopping for better professionalism
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    print("Starting training...")
    model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=50, 
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    model.save("models/gender_recognition_model.h5")
    joblib.dump(scaler, "models/scaler.joblib")
    print("Model and scaler saved to models/")

if __name__ == "__main__":
    train()
