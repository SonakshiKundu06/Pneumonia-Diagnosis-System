from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from PIL import Image
import base64
import io
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pneumonia Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "trained_pneumonia_disease_model.keras"
logger.info(f"Loading model from {MODEL_PATH}...")

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

IMAGE_SIZE = (128, 128)
CLASS_LABELS = ["NORMAL", "PNEUMONIA"]

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Open image and convert to RGB
        img = Image.open(io.BytesIO(image)).convert("RGB")
        # Resize image
        img = img.resize(IMAGE_SIZE)
        # Convert to array and preprocess
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

def encode_image_to_base64(img):
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Pneumonia Detection API is running. Use /predict endpoint to analyze chest X-rays."}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Predict whether a chest X-ray shows signs of pneumonia
    
    - **image**: Chest X-ray image file
    
    Returns prediction class and confidence score
    """
    try:
        logger.info(f"Received image: {image.filename}")
        
        # Read image content
        contents = await image.read()
        
        # Preprocess the image
        img_array, original_img = preprocess_image(contents)
        
        # Make prediction
        logger.info("Running prediction")
        predictions = model.predict(img_array)
        class_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_LABELS[class_index]
        confidence = float(np.max(predictions[0]))
        
        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
        
        # Encode image to base64
        base64_image = encode_image_to_base64(original_img)
        
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),  # Convert to percentage
            "image_base64": base64_image
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)