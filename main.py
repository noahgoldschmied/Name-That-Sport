import streamlit as st
import tensorflow as tf
import wikipediaapi
import plotly.graph_objects as go
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


st.set_page_config(page_title="Name That Sport")  # Set tab title

# this list holds all of the classes that the model has, will help with guess indexing
classes = ['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barrel racing',
           'baseball', 'basketball', 'baton twirling', 'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing',
           'bull riding', 'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket', 'croquet',
           'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men', 'figure skating pairs',
           'figure skating women', 'fly fishing', 'football', 'formula 1 racing', 'frisbee', 'gaga', 'giant slalom',
           'golf', 'hammer throw', 'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping',
           'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing', 'ice yachting',
           'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling', 'luge', 'motorcycle racing', 'mushing',
           'nascar racing', 'olympic wrestling', 'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo',
           'pommel horse', 'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby',
           'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping', 'sky surfing', 'skydiving',
           'snow boarding', 'snowmobile racing', 'speed skating', 'steer wrestling', 'sumo wrestling', 'surfing',
           'swimming', 'table tennis', 'tennis', 'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars',
           'volleyball', 'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball', 'wheelchair racing',
           'wingsuit flying']
classes = [name.encode('utf-8').decode('utf-8') for name in classes]


# Helper Functions to run the streamlit site

# load model, set cache to prevent reloading
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(f"Models/Sport_model_final.h5")
    return model


# image preprocessing
def load_image(image):
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = img / 255.0  # Normalize to [0, 1]
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return tf.cast(img, tf.float32)  # Change to float32

# image classifying
def classify_image(img: bytes, model: tf.keras.Model) -> pd.DataFrame:
    # Preprocess the image
    image = load_image(img)

    try:
        pred_probs = model.predict(image)
    except Exception as e:
        st.write("Error during model prediction:", e)
        raise

    # Get the indices of the top 3 predicted classes
    top_3_indices = sorted(pred_probs.argsort()[0][-3:][::-1])

    # Compute the probabilities for the top 3 predictions
    values = pred_probs[0][top_3_indices] * 100
    labels = [classes[i] for i in top_3_indices]

    # Create a DataFrame to store the top 3 predictions and their probabilities
    prediction_df = pd.DataFrame({
        "Sport Name": labels,
        "Probability": values,
    })

    # Sort the DataFrame by Probability
    return prediction_df.sort_values("Probability", ascending=False)


# Main Streamlit code

st.title("Noah Goldschmied's 'Name That Sport' App!")
st.caption("Welcome to my Name That Sport app")
st.caption("The app will take an image of a sport that you upload, and tell you what it thinks that sport is")
st.caption("The model has about an 84% accuracy, so please don't be alarmed if it guesses wrong")
st.caption("The model also does better with smaller images, so if you want better results, upload images with less pixels.")

# Get image URL from user
file = st.file_uploader(label="Upload an image of a sport.",
                        type=["JPEG", "PNG", "GIF", "BMP"])

if not file:
    image = None  # set to None because it doesn't exist yet
    pred_button = st.button("Guess that sport!", disabled=True, help="Upload an image to make a prediction")
    st.stop()
else:
    image = file.read()
    # Center the image using st.columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Display image in center column
        st.image(image, use_column_width="auto", width="50%")
    pred_button = st.button("Guess that sport!")

if pred_button:
    # Perform image classification and obtain prediction, confidence, and DataFrame
    with st.spinner("Loading Image Classification Model..."):
        model = load_model()

    with st.spinner("Classifying Image..."):
        df = classify_image(image, model)

    top_prediction_row = df.iloc[0]
    # Display the prediction and confidence
    st.success(
        f'Predicted Sport: **{top_prediction_row["Sport Name"].title()}** Confidence: {top_prediction_row["Probability"]:.2f}%')  # add something with scientific name here

    # create list for y-axis of plot which displays the name of the bird and links to the wikipedia page
    #y_axis_wiki_namelist = []
    #for index, row in df.iterrows():
    #    label = row["Sport Name"]
    #    current_link = row["Wikipedia Link"]
    #    y_axis_wiki_namelist.append(f'<a href="{current_link}" target="_blank">{label}</a>')

    fig = go.Figure(data=[
        go.Bar(
            x=df["Probability"],
            y=df["Sport Name"],
            orientation="h",
            text=df["Probability"].apply(lambda x: f"{x:.2f}%"),
            textposition="auto",
            marker=dict(color="lightgray"),
        )
    ])

    fig.update_layout(
        title="Sport Name",
        xaxis_title="Probability",
        yaxis_title="Sport",
        width=600,
        height=400,
        dragmode=False,
    )

    st.plotly_chart(fig)