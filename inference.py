import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Define the classes
classes = [
    "Adenocarcinoma Chest Lung Cancer",
    "Large cell carcinoma Lung Cancer",
    "NO Lung Cancer/ NORMAL",
    "Squamous cell carcinoma Lung Cancer"
]

def load_model_and_predict(image_path, model_name):
    # Load the pre-trained model from the specified model name
    model = tf.keras.models.load_model(f'Models/{model_name}')
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Make predictions
    predictions = model.predict(x)
    predicted_class_index = np.argmax(predictions)

    # Determine if cancerous and identify cancer type if applicable
    if predicted_class_index in [0, 1, 3]:
        is_cancerous = "Cancerous"
        cancer_type = classes[predicted_class_index]
    else:
        is_cancerous = "NO Lung Cancer/ NORMAL"
        cancer_type = "Normal"
    
    return f"{is_cancerous}|{cancer_type}"
