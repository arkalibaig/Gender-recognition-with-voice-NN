# Gender Recognition with Voice

This project predicts a speaker's gender based on their voice features using a neural network built in TensorFlow/Keras.

## Dataset

- Kaggle Voice Gender dataset: https://www.kaggle.com/datasets/primaryobjects/voicegender

## Model Overview

- A simple feedforward neural network trained on preprocessed audio features.
- Built using Keras with the following architecture:

```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

Limitations

Currently, the model doesn't work with live audio input from a microphone. It only predicts based on dataset samples.
File

    Gender_Recognition_by_voice.ipynb — the complete training and evaluation notebook.


---

