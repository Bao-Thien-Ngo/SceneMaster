import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
from utils import load_and_prep_image, classes_and_models, update_logger, predict_custom_trained_model_sample
import base64

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mystic-span-398721-0ffd73e42944.json"
PROJECT = "mystic-span-398721"
REGION = "us-central1"
endpoint_id = '4346547585482227712'
st.title("Welcome to Image Scene Classification")
st.header("Identify what's in your scene images!")


@st.cache  # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    image = load_and_prep_image(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    prediction = predict_custom_trained_model_sample(PROJECT,endpoint_id, image)
    pred_class = class_names[tf.argmax(prediction[0])]
    pred_conf = tf.reduce_max(prediction[0])

    return image, pred_class, pred_conf


# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (7 scene classes)",
     "Model 2 (15 scene classes)",
     "Model 3 (30 scene classes)")
)

# Model choice logic
if choose_model == "Model 1 (7 scene classes)":
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]

# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose {MODEL}, these are the scene classes it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of scene",
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(
        session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.3f}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=session_state.image,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct label be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))


