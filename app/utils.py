# Utils for preprocessing data etc 
import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import base64

base_classes = ['airplane',
                'beach',
                'bridge',
                'church',
                'harbor',
                'forest',
                'meadow']

classes_and_models = {
    "model_1": {
        "classes": base_classes,
        "model_name": "model_1_7_classes"
    }
}


def predict_custom_trained_model_sample(
        project: str,
        endpoint_id: str,
        instances: Union[Dict, List[Dict]],
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances_list = instances.numpy().tolist()  # {"data": instances.numpy().tolist()}
    instances = [
        json_format.ParseDict(instances_list, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # return response['predictions']
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

    return predictions


# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, rescale=False):
    # Reads in an image from filename, turns it into a tensor, and reshapes into(224, 224, 3).
    img = tf.io.decode_image(filename, channels=3)  # make sure there's 3 colour channels (for PNGs)
    img = tf.image.resize(img, [img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    if rescale:
        return img / 255.
    else:
        return img


def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
    # Function for tracking feedback given in-app, updates and returns logger dictionary.
    logger = {
        "image": image,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }
    return logger
