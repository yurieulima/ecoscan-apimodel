from fastapi import UploadFile, HTTPException
from services.model_service import predict
from PIL import Image
import numpy as np
import io

IMAGE_SIZE = (224, 224)

async def process_image(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are allowed")
    


    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 127.5 - 1
    image_array = np.expand_dims(image_array, axis=0)

    predicted_label, confidence, predicted_index = predict(image_array)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "shape": image_array.shape,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "predicted_index": int(predicted_index)
    }