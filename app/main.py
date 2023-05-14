from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from urllib.request import urlretrieve
import numpy as np
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import librosa

from fastapi import FastAPI


def initialize_weights(shape):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


model_path = "app/model.h5"

with custom_object_scope({'initialize_weights': initialize_weights, 'initialize_bias': initialize_bias}):
    model = load_model("app/model.h5")


def get_spectrogram_image(waveform, sr=22050):
    """
        Transforms a 'waveform' into a 'spectrogram image', adding padding if needed.
    """
    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=1024, frame_step=256, fft_length=1024)
    spectrogram = tf.abs(spectrogram)

    num_spectrogram_bins = spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrogram = tf.tensordot(
        spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)

    sample = tf.image.resize(mel_spectrogram, [224, 512])
    sample = tf.image.grayscale_to_rgb(sample)
    return sample


def process_audio(file_path):
    """
    Load an audio file and transform it into a spectrogram image.
    """
    sound, sample_rate = librosa.load(file_path)
    audio_trimmed, _ = librosa.effects.trim(sound, top_db=20)
    spectrogram_image = get_spectrogram_image(audio_trimmed)
    spectrogram_image = tf.expand_dims(spectrogram_image, 0)

    return spectrogram_image


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/predict")
async def predict(ref_url: str = Form(...), file: UploadFile = File(...)):
    # Fetch and save the audio file from the URL
    audio_name = ref_url.split("/")[-1]
    audio_path = f"temp/{audio_name}"
    urlretrieve(ref_url, audio_path)

    # Process the audio files
    try:
        audio_trimmed1 = process_audio(open(audio_path, 'rb').read())
        audio_trimmed2 = process_audio(await file.read())

        prediction = model.predict([audio_trimmed1, audio_trimmed2])
        os.remove(audio_path)
        prediction_score = np.asscalar(prediction[0][0])
        # return {"prediction": np.asscalar(prediction[0][0])}
        if prediction_score <= 0.25:
            return {"prediction": "1"}
        elif 0.25 < prediction_score <= 0.5:
            return {"prediction": "2"}
        else:
            return {"prediction": "3"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
