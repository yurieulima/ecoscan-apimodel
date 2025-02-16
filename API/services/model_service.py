from keras.api.models import load_model
import numpy as np

model = load_model("models/EcoScanModelV2.keras")

CLASS_MAPING = {
    0: "papelao",
    1: "vidro",
    2: "metal",
    3: "papel",
    4: "plastico",
    5: "lixo"
}

def predict(image_array: np.ndarray):

    prediction = model.predict(image_array)

    predicted_index = prediction.argmax(axis=1)[0] 

    predicted_label = CLASS_MAPING[predicted_index]

    confidence = prediction[0][predicted_index]

    return predicted_label, confidence, predicted_index