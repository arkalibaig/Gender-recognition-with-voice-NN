import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
import joblib

def load_and_preprocess_data(filepath):
    """Loads dataset, removes duplicates, and performs stratified scaling."""
    df = pd.read_csv(filepath)
    
    print("\n--- Data Diagnostics ---")
    print(f"Initial dataset shape: {df.shape}")
    print("Class distribution (raw):")
    print(df["label"].value_counts())
    
    # Check for duplicates which can cause inflated accuracy
    initial_count = len(df)
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows found: {duplicates}")
    
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")
    print("Class distribution (after removing duplicates):")
    print(df["label"].value_counts())
    
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    
    # Use stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test), scaler

def build_model(input_shape):
    """Builds a regularized Sequential ANN model."""
    # Reduced capacity and added L2 regularization to prevent memorization
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    return model

def train():
    """Enhanced training pipeline with regularization and learning rate scheduling."""
    data_path = "data/vocal_gender_features.csv"
    if not os.path.exists(data_path):
        # Try finding it in the parent directory if run from scripts/
        data_path = "../data/vocal_gender_features.csv"
        if not os.path.exists(data_path):
            print(f"Error: Dataset not found.")
            return

    (X_train, X_test, y_train, y_test), scaler = load_and_preprocess_data(data_path)
    
    model = build_model(X_train.shape[1])
    
    # More robust callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\nStarting robust training...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=100, 
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # Classification Report
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    model.save(os.path.join(model_dir, "gender_recognition_model.h5"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    print(f"Model and scaler saved to {model_dir}/")

if __name__ == "__main__":
    train()
