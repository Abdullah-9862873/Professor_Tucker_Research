#!/usr/bin/env python
"""FastAPI service for Safety Boundary Detection."""

import torch
import io
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import base64
import cv2

app = FastAPI(
    title="Safety Boundary Detection API",
    description="API for detecting safety boundary violations (safe, subtle, obvious)",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = Path('config/config.yaml')
MODEL_PATH = Path('backend/models/latest_checkpoint.pth')
BEST_MODEL_PATH = Path('backend/models/best_model.pth')

model = None
cfg = None
device = None

CLASS_NAMES = ['safe', 'subtle', 'obvious']
CLASS_COLORS = {
    'safe': (0, 255, 0),
    'subtle': (255, 255, 0),
    'obvious': (255, 0, 0)
}

def load_model():
    """Load model and configuration."""
    global model, cfg, device
    
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    
    from safety_detector import SafetyBoundaryDetector
    
    device = torch.device('cpu')
    
    checkpoint_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else MODEL_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    model = SafetyBoundaryDetector(
        base_model=cfg.get('model', {}).get('base_model', 'yolov8n.pt'),
        num_classes=3
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    safety_head_keys = [k for k in state_dict.keys() if 'safety_head' in k]
    if safety_head_keys:
        model_state = model.state_dict()
        for key in safety_head_keys:
            if key in model_state:
                model_state[key] = state_dict[key]
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded {len(safety_head_keys)} safety_head weights")
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    """Detect safety boundaries in uploaded image."""
    if model is None:
        load_model()
    
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        img_np = np.array(img)
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # (C,H,W) -> (N,C,H,W) for interpolate
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(320, 320), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            result = model(img_tensor)
        
        safety_scores = result['safety_scores']
        if safety_scores.dim() == 3:
            safety_scores = safety_scores.squeeze(1)
        
        probs = torch.softmax(safety_scores, dim=1)[0]
        pred_class = torch.argmax(probs, dim=0).item()
        confidence = probs[pred_class].item()
        
        detections = []
        if 'detections' in result and result['detections'] is not None:
            dets = result['detections']
            if hasattr(dets, 'boxes') and dets.boxes is not None:
                boxes = dets.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    detections.append({
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    })
        
        response = {
            'prediction': CLASS_NAMES[pred_class],
            'class_id': int(pred_class),
            'confidence': float(confidence),
            'probabilities': {
                'safe': float(probs[0]),
                'subtle': float(probs[1]),
                'obvious': float(probs[2])
            },
            'detections': detections,
            'image_size': {'width': img.width, 'height': img.height}
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_with_visualization")
async def detect_with_viz(file: UploadFile = File(...)) -> JSONResponse:
    """Detect and return visualization image."""
    if model is None:
        load_model()
    
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        img_np = np.array(img).copy()
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # (C,H,W) -> (N,C,H,W) for interpolate
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(320, 320), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            result = model(img_tensor)
        
        safety_scores = result['safety_scores']
        if safety_scores.dim() == 3:
            safety_scores = safety_scores.squeeze(1)
        
        probs = torch.softmax(safety_scores, dim=1)[0]
        pred_class = torch.argmax(probs, dim=0).item()
        
        label = CLASS_NAMES[pred_class]
        color = CLASS_COLORS[label]
        
        cv2.putText(
            img_np,
            f"{label.upper()}: {probs[pred_class].item():.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )
        
        for i, prob in enumerate(probs.tolist()):
            text = f"{CLASS_NAMES[i]}: {prob:.2f}"
            cv2.putText(
                img_np,
                text,
                (10, 60 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        _, img_encoded = cv2.imencode('.png', img_np)
        img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        return JSONResponse(content={
            'prediction': CLASS_NAMES[pred_class],
            'class_id': int(pred_class),
            'visualization': f"data:image/png;base64,{img_b64}"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
async def get_classes() -> JSONResponse:
    """Get available classes."""
    return JSONResponse(content={
        'classes': CLASS_NAMES,
        'class_ids': {name: i for i, name in enumerate(CLASS_NAMES)}
    })

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Safety Boundary Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This info",
            "/health": "Health check",
            "/detect": "POST image for detection",
            "/classes": "Available classes"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)