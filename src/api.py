"""
src/api.py
----------
FastAPI gateway for Tier-1 (ONNX) + Tier-2 (Gemini) + XAI (Grad-CAM).
"""

import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
import io

from src.fallback import SpiceFallbackEngine
from src.explain import init_cam_model, generate_gradcam, encode_image_base64

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ONNX_PATH = MODELS_DIR / "model.onnx"
PT_PATH = MODELS_DIR / "best_model.pth"
STATIC_DIR = PROJECT_ROOT / "static"

class SystemState:
    ort_session = None
    cam_model = None
    class_indices = None
    idx_to_class = None
    knowledge_base = None
    fallback_engine = None
    device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing System State...")
    SystemState.device = torch.device("cpu")
    
    # 1. Load ONNX Runtime Session
    print(f"Loading ONNX Session from {ONNX_PATH}...")
    SystemState.ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    # 2. Load PyTorch model for Grad-CAM
    print(f"Loading PyTorch CAM Model from {PT_PATH}...")
    SystemState.cam_model = init_cam_model(PT_PATH, SystemState.device, num_classes=14)
    
    # 3. Load Data & Knowledge Base
    class_idx_path = PROJECT_ROOT / "data" / "class_indices.json"
    kb_path = PROJECT_ROOT / "data" / "spice_knowledge.json"
    
    with open(class_idx_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        SystemState.class_indices = data["class_to_idx"]
        SystemState.idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
        
    with open(kb_path, "r", encoding="utf-8") as f:
        SystemState.knowledge_base = json.load(f)
        
    # 4. Initialize Fallback Engine
    print("Initializing SpiceFallbackEngine...")
    SystemState.fallback_engine = SpiceFallbackEngine()
    
    yield
    print("Shutting down...")

app = FastAPI(title="Spice Verification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Softmax helper
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

import torchvision.transforms as transforms

# Preprocessing for ONNX (Strict PyTorch Parity)
def preprocess_image(image: Image.Image) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Convert to RGB, apply transforms, add batch dimension, and convert to numpy
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor.numpy()

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "onnx_loaded": SystemState.ort_session is not None,
        "cam_loaded": SystemState.cam_model is not None,
        "fallback_ready": SystemState.fallback_engine is not None
    }

@app.post("/api/v1/classify")
async def classify(
    image: UploadFile = File(...),
    threshold: float = Form(0.75),
    force_fallback: bool = Form(False)
):
    import time
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {str(e)}"})

    # Run Tier-1 (ONNX)
    start_time = time.perf_counter()
    input_tensor = preprocess_image(img)
    ort_inputs = {SystemState.ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = SystemState.ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    probs = softmax(logits)[0]
    top_indices = np.argsort(probs)[::-1]
    
    top_1_idx = top_indices[0]
    top_1_prob = float(probs[top_1_idx])
    top_1_class = SystemState.idx_to_class[top_1_idx]
    
    top_3 = []
    for idx in top_indices[:3]:
        cls_name = SystemState.idx_to_class[idx]
        disp_name = SystemState.knowledge_base.get(cls_name, {}).get("display_name", cls_name)
        top_3.append({
            "class_id": cls_name,
            "display_name": disp_name,
            "probability": float(probs[idx])
        })

    # Gateway Routing
    if top_1_prob >= threshold and not str(force_fallback).lower() == "true":
        kb_entry = SystemState.knowledge_base.get(top_1_class, {})
        return {
            "tier": "TIER_1_ONNX_LOCAL",
            "prediction": {
                "class_id": top_1_class,
                "display_name": kb_entry.get("display_name", top_1_class),
                "confidence": top_1_prob,
            },
            "top_3": top_3,
            "metadata": kb_entry,
            "latency_ms": inference_time_ms
        }
    else:
        # Tier-2
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        fallback_start = time.perf_counter()
        try:
            fallback_res = SystemState.fallback_engine.identify_unknown_image(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        fallback_time_ms = round((time.perf_counter() - fallback_start) * 1000, 2)
                
        return {
            "tier": "TIER_2_GEMINI_FALLBACK",
            "fallback_response": fallback_res,
            "tier_1_context": {
                "top_1_class": top_1_class,
                "top_1_prob": top_1_prob,
                "top_3": top_3
            },
            "latency_ms": fallback_time_ms
        }

@app.post("/api/v1/explain")
async def explain(
    image: UploadFile = File(...),
    target_class_idx: Optional[str] = Form(None)
):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {str(e)}"})
        
    actual_idx = None
    if target_class_idx is not None:
        if target_class_idx in SystemState.class_indices:
            actual_idx = SystemState.class_indices[target_class_idx]
        else:
            try:
                actual_idx = int(target_class_idx)
            except ValueError:
                actual_idx = None

    blended_bgr, pred_idx, conf = generate_gradcam(
        SystemState.cam_model, 
        img, 
        SystemState.device, 
        actual_idx
    )
    
    b64_image = encode_image_base64(blended_bgr)
    
    pred_class = SystemState.idx_to_class[pred_idx]
    disp_name = SystemState.knowledge_base.get(pred_class, {}).get("display_name", pred_class)
    
    return {
        "class_id": pred_class,
        "display_name": disp_name,
        "confidence": conf,
        "heatmap_base64": b64_image
    }

STATIC_DIR.mkdir(exist_ok=True, parents=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
