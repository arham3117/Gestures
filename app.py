"""
GestureFlow API - Hand Gesture Recognition Service
FastAPI application for serving gesture predictions
"""

import io
import base64
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tensorflow as tf

from config import GESTURES, MODEL_IMAGE_SIZE


# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    print("Loading gesture recognition model...")
    model = tf.keras.models.load_model("models/gesture_model.h5")
    print("Model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="GestureFlow API",
    description="Real-time hand gesture recognition using CNN",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    image: str  # Base64 encoded image


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint."""
    gesture: str
    confidence: float
    all_predictions: dict[str, float]


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    model_loaded: bool


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model inference.

    Must match predict_live.py preprocessing:
    1. Convert to grayscale (cv2.cvtColor BGR2GRAY)
    2. Apply GaussianBlur (7,7)
    3. Resize to 64x64
    4. Normalize to 0-1
    """
    from PIL import ImageFilter

    # Open image (comes as RGB from canvas)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale
    image = image.convert("L")

    # Apply Gaussian blur - OpenCV (7,7) kernel is roughly PIL radius=2
    image = image.filter(ImageFilter.GaussianBlur(radius=2))

    # Resize to 64x64 using BILINEAR (similar to cv2 default)
    image = image.resize(MODEL_IMAGE_SIZE, Image.Resampling.BILINEAR)

    # Convert to numpy array and normalize to 0-1
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Add batch and channel dimensions: (1, 64, 64, 1)
    img_array = img_array.reshape(1, MODEL_IMAGE_SIZE[0], MODEL_IMAGE_SIZE[1], 1)

    return img_array


# Serve static files (frontend)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for container orchestration."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict gesture from base64 encoded image.

    Args:
        request: Contains base64 encoded image string

    Returns:
        Predicted gesture, confidence, and all class probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        # Preprocess image
        processed_image = preprocess_image(image_bytes)

        # Get prediction
        predictions = model.predict(processed_image, verbose=0)[0]

        # Get predicted class and confidence
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        gesture = GESTURES[predicted_idx]

        # Create prediction dictionary for all classes
        all_predictions = {
            GESTURES[i]: float(predictions[i])
            for i in range(len(GESTURES))
        }

        return PredictionResponse(
            gesture=gesture,
            confidence=confidence,
            all_predictions=all_predictions
        )

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/gestures")
async def list_gestures():
    """List all recognized gestures."""
    return {"gestures": GESTURES, "count": len(GESTURES)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
