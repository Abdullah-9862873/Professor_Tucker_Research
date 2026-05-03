#!/usr/bin/env python
"""Evaluation module for SafetyBoundaryDetector with metrics, confusion matrix, CI, and baseline comparison."""

import torch
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    classification_report
)
from sklearn.utils import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ['safe', 'subtle', 'obvious']

def evaluate_predictions(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, per-class F1, macro-F1, and accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        targets, preds, average='macro', labels=[0, 1, 2], zero_division=0
    )[2]
    accuracy = accuracy_score(targets, preds)
    
    return {
        'precision_safe': float(precision[0]),
        'precision_subtle': float(precision[1]),
        'precision_obvious': float(precision[2]),
        'recall_safe': float(recall[0]),
        'recall_subtle': float(recall[1]),
        'recall_obvious': float(recall[2]),
        'f1_safe': float(f1[0]),
        'f1_subtle': float(f1[1]),
        'f1_obvious': float(f1[2]),
        'macro_f1': float(macro_f1),
        'accuracy': float(accuracy),
    }

def compute_confusion_matrix(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for 3-class safety classification."""
    return confusion_matrix(targets, preds, labels=[0, 1, 2])

def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """Plot and save confusion matrix as PNG."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Safety Boundary Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def compute_confidence_intervals(
    preds: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 1000,
    metric: str = 'macro_f1'
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.
    
    Returns: (lower_bound, mean, upper_bound) at 95% CI
    """
    scores = []
    for _ in range(n_bootstrap):
        indices = resample(np.arange(len(preds)), random_state=None)
        pred_sample = preds[indices]
        target_sample = targets[indices]
        
        if metric == 'macro_f1':
            f1 = precision_recall_fscore_support(
                target_sample, pred_sample, average='macro', zero_division=0
            )[2]
            scores.append(f1)
        elif metric == 'accuracy':
            scores.append(accuracy_score(target_sample, pred_sample))
    
    scores = np.array(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    mean = np.mean(scores)
    return float(lower), float(mean), float(upper)

def evaluate_model(model, dataloader, device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Run full evaluation on a dataloader.
    
    Returns: (metrics_dict, predictions, targets)
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            out_dict = model(images)
            logits = out_dict["safety_scores"]
            if logits.dim() == 3:
                logits = logits.squeeze(1)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = evaluate_predictions(preds, targets)
    
    return metrics, preds, targets

def generate_evaluation_report(
    metrics: Dict[str, float],
    cm: np.ndarray,
    output_dir: Path,
    model_name: str = "SafetyBoundaryDetector"
) -> None:
    """Generate markdown report with metrics and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')
    
    report = f"""# Evaluation Report: {model_name}

## Overall Metrics
| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Macro F1 | {metrics['macro_f1']:.4f} |

## Per-Class Metrics
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Safe | {metrics['precision_safe']:.4f} | {metrics['recall_safe']:.4f} | {metrics['f1_safe']:.4f} |
| Subtle | {metrics['precision_subtle']:.4f} | {metrics['recall_subtle']:.4f} | {metrics['f1_subtle']:.4f} |
| Obvious | {metrics['precision_obvious']:.4f} | {metrics['recall_obvious']:.4f} | {metrics['f1_obvious']:.4f} |

## Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---
*Report generated by safety_evaluator.py*
"""
    
    (output_dir / 'evaluation_report.md').write_text(report)
    print(f'Report saved to {output_dir / "evaluation_report.md"}')

def evaluate_baseline_yolo(
    val_loader,
    device,
    base_model: str = "yolov8n.pt"
) -> Dict[str, float]:
    """Run vanilla YOLOv8 on validation set for baseline comparison."""
    from ultralytics import YOLO
    
    print(f"Evaluating baseline YOLOv8 ({base_model})...")
    yolo = YOLO(base_model)
    
    all_preds = []
    all_targets = []
    
    for batch in val_loader:
        images = batch['image'].to(device)
        targets = batch['label'].cpu().numpy()
        
        results = yolo(images, verbose=False)
        
        for i, result in enumerate(results):
            pred_class = 0
            if len(result.boxes) > 0:
                boxes = result.boxes
                confidences = boxes.conf.cpu().numpy()
                if len(confidences) > 0:
                    if max(confidences) > 0.7:
                        pred_class = 2
                    elif max(confidences) > 0.3:
                        pred_class = 1
            
            all_preds.append(pred_class)
            all_targets.append(targets[i])
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    return evaluate_predictions(preds, targets)

def cross_validate(
    model_class,
    dataset,
    k: int = 5,
    epochs: int = 3,
    batch_size: int = 8,
    device: str = 'cpu'
) -> List[Dict[str, float]]:
    """Run k-fold cross-validation.
    
    Returns list of metric dicts, one per fold.
    """
    from sklearn.model_selection import KFold
    from torch.utils.data import Subset
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Training fold {fold + 1}/{k}...")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        model = model_class(num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                out = model(images)
                logits = out["safety_scores"].squeeze(1) if out["safety_scores"].dim() == 3 else out["safety_scores"]
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        
        metrics, _, _ = evaluate_model(model, val_loader, device)
        fold_metrics.append(metrics)
        
        del model, optimizer
    
    return fold_metrics

def main():
    """CLI for running evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Safety Boundary Detector')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='reports', help='Output directory')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    from safety_detector import SafetyBoundaryDetector
    from safety_data_loader import create_dataloader
    
    device = torch.device('cpu')
    
    model = SafetyBoundaryDetector(num_classes=3)
    checkpoint = torch.load(args.checkpoint, map_location=device)
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
    
    dataset_path = cfg['dataset']['coco_path']
    safety_bounds = cfg.get('safety_boundaries', {})
    batch_size = cfg['dataset'].get('batch_size', 16)
    
    val_loader, _ = create_dataloader(
        dataset_path=dataset_path,
        safety_boundaries=safety_bounds,
        split='val',
        batch_size=batch_size,
        max_samples=50,
        num_workers=0,
    )
    
    metrics, preds, targets = evaluate_model(model, val_loader, device)
    cm = compute_confusion_matrix(preds, targets)
    
    lower, mean, upper = compute_confidence_intervals(preds, targets, metric='macro_f1')
    metrics['macro_f1_ci_lower'] = lower
    metrics['macro_f1_ci_mean'] = mean
    metrics['macro_f1_ci_upper'] = upper
    
    output_dir = Path(args.output)
    generate_evaluation_report(metrics, cm, output_dir)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Macro F1: {metrics['macro_f1']:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

if __name__ == '__main__':
    main()