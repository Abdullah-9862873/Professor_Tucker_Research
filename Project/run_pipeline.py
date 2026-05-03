#!/usr/bin/env python
"""CLI entry point for the Safety Boundary Detection pipeline.

Usage:
    python run_pipeline.py --phase prepare   # Download & prepare data
    python run_pipeline.py --phase train      # Train model
    python run_pipeline.py --phase eval       # Evaluate model
    python run_pipeline.py --phase infer      # Single image inference
    python run_pipeline.py --phase serve      # Start API server
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))


def prepare_data():
    """Download and prepare COCO dataset."""
    print("=== DATA PREPARATION ===")
    print("COCO data should already be in: data/annotations/")
    print("If not, run: python scripts/download_coco.py")
    print("Safety annotations are applied automatically during training.")


def train_model(config_path: str = "config/config.yaml"):
    """Train the model."""
    print("=== TRAINING ===")
    from safety_trainer import main
    main()


def evaluate_model(checkpoint: str = "backend/models/best_model.pth"):
    """Run evaluation."""
    print("=== EVALUATION ===")
    from safety_evaluator import main as eval_main
    sys.argv = ["safety_evaluator.py", "--checkpoint", checkpoint]
    eval_main()


def run_inference(image_path: str, checkpoint: str = "backend/models/best_model.pth"):
    """Run single image inference."""
    print(f"=== INFERENCE: {image_path} ===")
    import torch
    from PIL import Image
    import yaml
    from safety_detector import SafetyBoundaryDetector
    
    cfg = yaml.safe_load(open("config/config.yaml"))
    device = torch.device("cpu")
    
    model = SafetyBoundaryDetector(num_classes=3)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = torch.nn.functional.resize(img_tensor, (320, 320)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        result = model(img_tensor)
    
    probs = torch.softmax(result['safety_scores'].squeeze(1), dim=1)[0]
    pred = torch.argmax(probs, dim=0).item()
    
    classes = ['safe', 'subtle', 'obvious']
    print(f"Prediction: {classes[pred]}")
    print(f"Confidence: {probs[pred].item():.2%}")
    print("Probabilities:")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {probs[i].item():.2%}")


def start_api():
    """Start the FastAPI server."""
    print("=== STARTING API SERVER ===")
    import uvicorn
    uvicorn.run("backend.src.api.safety_api:app", host="0.0.0.0", port=8000)


def start_streamlit():
    """Start the Streamlit frontend."""
    print("=== STARTING STREAMLIT FRONTEND ===")
    import subprocess
    subprocess.run(["streamlit", "run", "frontend/app.py"])


def main():
    parser = argparse.ArgumentParser(description="Safety Boundary Detection Pipeline")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["prepare", "train", "eval", "infer", "serve", "ui"],
        required=True,
        help="Pipeline phase to run"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path (for eval/infer)")
    parser.add_argument("--image", type=str, help="Image path (for infer)")
    
    args = parser.parse_args()
    
    if args.phase == "prepare":
        prepare_data()
    elif args.phase == "train":
        train_model(args.config)
    elif args.phase == "eval":
        checkpoint = args.checkpoint or "backend/models/best_model.pth"
        evaluate_model(checkpoint)
    elif args.phase == "infer":
        if not args.image:
            print("Error: --image required for infer phase")
            sys.exit(1)
        checkpoint = args.checkpoint or "backend/models/best_model.pth"
        run_inference(args.image, checkpoint)
    elif args.phase == "serve":
        start_api()
    elif args.phase == "ui":
        start_streamlit()


if __name__ == "__main__":
    main()