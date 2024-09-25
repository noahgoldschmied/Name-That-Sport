import streamlit as st
import tensorflow as tf

from tf.keras.models import load_model

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

st.header("Image Recognition App")
st.caption("Upload an image. ")
st.caption("The application will infer the one label out of 4 labels: 'Cloudy', 'Desert', 'Green_Area', 'Water'.")
st.caption("Warning: Do not click Recognize button before uploading image. It will result in error.")

# Load the model
model = load_model("Models/Sport_model_final.h5")
