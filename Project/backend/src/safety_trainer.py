#!/usr/bin/env python
"""Training script for SafetyBoundaryDetector with proper validation and checkpointing."""

import torch
import numpy as np
import random
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Import our components
from safety_detector import SafetyBoundaryDetector
from training_images_loader import create_training_dataloader
from utils import seed_everything, parse_config

CHECKPOINT_DIR = Path('backend/models')
LATEST_CHECKPOINT = CHECKPOINT_DIR / 'latest_checkpoint.pth'
BEST_MODEL = CHECKPOINT_DIR / 'best_model.pth'
STATE_FILE = CHECKPOINT_DIR / 'training_state.json'

def get_checkpoint_path(epoch: Optional[int] = None) -> Path:
    """Return path for checkpoint. If epoch provided, save per-epoch checkpoint."""
    if epoch is not None:
        return CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
    return LATEST_CHECKPOINT

def save_checkpoint(model, optim, epoch: int, best_f1: float, cfg: dict):
    """Save checkpoint with model, optimizer, epoch, and best F1."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'epoch': epoch,
        'best_f1': best_f1,
        'config': cfg
    }
    torch.save(checkpoint, LATEST_CHECKPOINT)
    with open(STATE_FILE, 'w') as f:
        json.dump({'current_epoch': epoch, 'best_f1': best_f1, 'last_updated': time.time()}, f)

def load_checkpoint(model, optim) -> Tuple[int, float]:
    """Load checkpoint if exists. Returns (start_epoch, best_f1)."""
    if not LATEST_CHECKPOINT.exists():
        return 1, 0.0
    try:
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        safety_head_keys = [k for k in state_dict.keys() if 'safety_head' in k]
        
        if safety_head_keys:
            model_state = model.state_dict()
            for key in safety_head_keys:
                if key in model_state:
                    model_state[key] = state_dict[key]
            model.load_state_dict(model_state, strict=False)
            print(f'✓ Loaded {len(safety_head_keys)} safety_head weights from checkpoint')
        
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f'✓ Resuming from epoch {start_epoch} (best F1: {best_f1:.3f})')
        return start_epoch, best_f1
    except Exception as e:
        print(f'Warning: Could not load checkpoint ({e}), starting fresh')
        return 1, 0.0

def save_best_model(model, epoch: int, f1: float):
    """Save best model if F1 improved."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'f1': f1
    }, BEST_MODEL)

def train_one_epoch(model, loader, optim, device):
    """Train one epoch and return average loss.

    Uses CrossEntropyLoss on the per‑image safety label (0=safe,1=subtle,2=obvious).
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    for batch in loader:
        # Move data to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optim.zero_grad()
        # Forward returns a dict; extract safety scores logits
        out_dict = model(images)
        logits = out_dict["safety_scores"].squeeze(1) if out_dict["safety_scores"].dim() == 3 else out_dict["safety_scores"]
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device) -> Dict[str, float]:
    """Evaluate model on a validation set.

    Returns macro‑averaged F1 score and overall accuracy for the
    three‑class safety classification (0=safe, 1=subtle, 2=obvious).
    """
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            out_dict = model(images)
            # Extract logits for safety scores
            logits = out_dict["safety_scores"].squeeze(1) if out_dict["safety_scores"].dim() == 3 else out_dict["safety_scores"]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    preds_logits = torch.cat(all_logits)
    true_labels = torch.cat(all_labels)
    # Predicted class is argmax over logits
    pred_classes = torch.argmax(preds_logits, dim=1).numpy()
    true_classes = true_labels.numpy()
    # Compute macro‑averaged F1 and accuracy
    f1 = f1_score(true_classes, pred_classes, average='macro')
    accuracy = accuracy_score(true_classes, pred_classes)
    return {'f1': f1, 'accuracy': accuracy}

def main():
    # Load config
    cfg_path = Path('config/config.yaml')
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    
    # Seed for reproducibility
    seed_everything(cfg.get('training', {}).get('seed', 42))
    
    # Create dataset and dataloader using the helper
    # Config keys expected: dataset.coco_path, safety_boundaries, training.batch_size, training.num_workers
    dataset_path = cfg['dataset']['coco_path']
    safety_bounds = cfg.get('safety_boundaries', {})
    batch_size = cfg['dataset'].get('batch_size', 16)
    num_workers = cfg['dataset'].get('num_workers', 0)
    max_train_samples = cfg['dataset'].get('max_train_samples', None)
    max_val_samples = cfg['dataset'].get('max_val_samples', 50)  # Small for fast eval
    loader, _ = create_training_dataloader(
        data_dir='training-images',
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    # Validation - use subset of training
    val_loader, _ = create_training_dataloader(
        data_dir='training-images',
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    
    # Create model and move to device
    device_str = cfg.get('training', {}).get('device', 'cpu')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print('⚠️ CUDA not available, falling back to CPU')
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f'Device: {device}')
    model = SafetyBoundaryDetector(
        base_model=cfg.get('model', {}).get('base_model', 'yolov8m.pt'),
        num_classes=cfg.get('model', {}).get('num_classes', 80)
    )
    
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['training'].get('learning_rate', 5e-4))
    
    total_epochs = cfg['training'].get('epochs', 30)
    num_batches = len(loader)
    
    start_epoch, best_f1 = load_checkpoint(model, optim)
    
    print(f'=== TRAINING STARTED ===')
    print(f'📊 Dataset: {num_batches * batch_size} training images | {len(val_loader) * batch_size} validation images')
    print(f'🎯 Training for {total_epochs} epochs (starting from epoch {start_epoch})')
    print(f'💾 Checkpoints will be saved to: {CHECKPOINT_DIR}')
    print('-' * 60)
    
    start_time = time.time()
    for epoch in tqdm(range(start_epoch, total_epochs + 1), desc='Training', ncols=80):
        epoch_start = time.time()
        
        loss = train_one_epoch(model, loader, optim, device)
        
        save_checkpoint(model, optim, epoch, best_f1, cfg)
        
        if epoch % 2 == 0:
            metrics = evaluate(model, val_loader, device)
            f1, acc = metrics['f1'], metrics['accuracy']
            
            if f1 > best_f1:
                best_f1 = f1
                save_best_model(model, epoch, f1)
                improvement = ' ⭐ NEW BEST!'
            else:
                improvement = ''
            
            tqdm.write(f'Epoch {epoch:02d}/{total_epochs} | Loss: {loss:.4f} | F1: {f1:.3f} | Acc: {acc:.3f}{improvement}')
        else:
            tqdm.write(f'Epoch {epoch:02d}/{total_epochs} | Loss: {loss:.4f}')
        
        epoch_time = time.time() - epoch_start
        avg_time = (time.time() - start_time) / (epoch - start_epoch + 1)
        remaining = avg_time * (total_epochs - epoch)
        tqdm.write(f'   ⏱️ Epoch time: {epoch_time:.1f}s | ETA: {remaining/60:.1f} min')
    
    print('-' * 60)
    print(f'✅ Training complete! Best F1: {best_f1:.3f}')
    print(f'📁 Checkpoints saved to: {CHECKPOINT_DIR}')

if __name__ == '__main__':
    main()