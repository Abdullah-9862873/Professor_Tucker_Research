# Implementation Plan: Safety Boundary Detection Analysis

## Objective
Create a Jupyter Notebook demonstrating:
1. Initial model performs well on Safe images but poorly on Subtle/Obvious
2. Retrain/fine-tune to improve Subtle/Obvious detection
3. Single comprehensive notebook file

---

## Dataset Summary
- **Safe**: 100 images (label=0) - No danger detected
- **Subtle**: 192 images (label=1) - Low danger score (~190)
- **Obvious**: 149 images (label=2) - High danger score (~370)
- **COCO val2017**: 5000 images (for augmentation/transfer learning)

---

## Notebook Structure

### Section 1: Setup & Imports
- Import PyTorch, YOLOv8, torchvision, matplotlib, etc.
- Set device (GPU/CPU)
- Define paths

### Section 2: Data Loading
- Load labeled dataset (Safe, Subtle, Obvious)
- Split into train/val/test (70/15/15)
- Analyze class distribution

### Section 3: Initial Model Training (Baseline)
- Use YOLOv8 or ResNet backbone
- Train on current dataset
- Evaluate per-class performance
- **Expected Result**: High accuracy on Safe, low on Subtle/Obvious

### Section 4: Performance Analysis
- Confusion matrix
- Per-class precision/recall/F1
- Visualize misclassifications
- **Key Finding**: Model biased toward Safe class

### Section 5: Data Augmentation & Retraining
- Augment Subtle/Obvious samples (flip, rotate, brightness)
- Apply class weighting to handle imbalance
- Fine-tune with lower learning rate
- Use COCO pretrained weights

### Section 6: Improved Model Evaluation
- Compare before/after metrics
- Show improvement in Subtle/Obvious detection
- Save final model

### Section 7: Conclusions
- Summary of findings
- Recommendations for further improvement

---

## Technical Approach (FIXED based on Persona Review)
1. **Baseline**: Use YOLOv8 features + classification head (NOT ResNet18)
2. **Analysis**: Run baseline FIRST, measure actual performance, THEN show problem
3. **Fix**: Class weighting + proper fine-tuning (freeze→unfreeze) + removed COCO contamination
4. **Validation**: K-fold CV + statistical significance + proper baselines (random, majority)

## Critical Fixes Applied:
- Remove COCO "Safe" labeling (was contaminating training data)
- Use YOLOv8 feature extraction backbone (matches original project)
- Add k-fold cross-validation for robustness
- Add statistical significance testing (confidence intervals)
- Proper stratified split (COCO only in train, not test/val)
- Add proper fine-tuning schedule (phase-based learning rates)

---

## Files Created
- `safety_detection_analysis.ipynb` - Main notebook (COMPLETED)
- `IMPLEMENTATION_PLAN.md` - This plan

## Notebook Contents (Full Implementation)
1. **Setup & Imports** - PyTorch, YOLO, sklearn, etc.
2. **Data Loading** - 441 labeled images (100 Safe, 192 Subtle, 149 Obvious)
3. **Baseline Training** - Shows problem: good Safe, poor Subtle/Obvious
4. **Performance Analysis** - Confusion matrix, per-class accuracy
5. **Improved Model** - Class weighting, lower LR, better scheduler
6. **COCO Augmentation** - Added 500 COCO images for more diverse training
7. **Results Comparison** - Before/after metrics visualization
8. **Conclusions** - Summary and recommendations

## Key Results (Expected when run)
- Baseline: Safe ~85%, Subtle ~45%, Obvious ~55%
- Improved: All classes ~65-75%
- Shows the problem and solution in one notebook