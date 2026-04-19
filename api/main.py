import io
import json

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image

from src.infer import predict_disease

# Initialize FastAPI with metadata for Swagger
app = FastAPI(
    title="Plant Disease API",
    description="An API to identify plant diseases from images.",
    version="1.0.0",
)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and mapping globally
try:
    model = torch.jit.load("convnext_scripted.pt", map_location=device)
    model.eval()

    with open("data/label_map.json") as f:
        label_map = json.load(f)
    # Ensure keys are handled correctly (mapping string indices to names)
    idx_to_disease = {int(v): k for k, v in label_map.items()}
except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None


@app.get("/", include_in_schema=False)
async def root():
    """Redirect users to the Swagger UI automatically."""
    return RedirectResponse(url="/docs")


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Upload an image of a plant leaf to identify potential diseases.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 1. Read and Preprocess
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 2. Run Inference
        disease_name = predict_disease(model, image, idx_to_disease, device=device)

        return {"disease": disease_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
