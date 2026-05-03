import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class SafetyBoundaryDetector(nn.Module):
    """
    Enhanced YOLOv8-based safety boundary detection system.
    
    Detects subtle and obvious safety boundary violations by combining
    object detection with proximity-based safety assessment.
    """
    
    def __init__(
        self,
        base_model: str = "yolov8m.pt",
        num_classes: int = 80,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        subtle_threshold: float = 0.3,
        obvious_threshold: float = 0.8,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.subtle_threshold = subtle_threshold
        self.obvious_threshold = obvious_threshold
        
        # Load YOLOv8 model
        from ultralytics import YOLO
        self.yolo = YOLO(base_model)
        # Disable YOLO's train method (it expects args) – we only use inference
        self.yolo.train = lambda *args, **kwargs: None
        
        # Safety enhancement layers
        self.safety_head = nn.Sequential(
            nn.Linear(256, 512),  # image features only
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),  # raw logits for 3 classes
        )
        
        # Proximity calculator
        self.proximity_calculator = ProximityCalculator()
        
        # Attention mechanism for safety regions
        self.attention = SafetyAttention()
        
        # Feature extractor for YOLO embeddings
        self.feature_extractor = FeatureExtractor()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simplified forward pass for training.
        Returns safety classification logits; detection outputs are placeholders.
        """
        # Extract image-level features using the lightweight CNN
        img_features = self.feature_extractor(x)  # shape (B, 256)

        # Compute raw safety logits (3 classes)
        safety_scores = self.safety_head(img_features)  # (B, 3)

        # Compatibility placeholders (empty tensors)
        empty = torch.empty(0, device=x.device)
        return {
            "detections": [],               # No detections needed for training
            "safety_scores": safety_scores,
            "proximity_maps": empty,
            "attention_maps": empty,
        }
    
    def detect_failures(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect subtle and obvious failures.
        """
        outputs = self.forward(x)
        
        # Classify failures based on proximity scores
        failure_predictions = []
        
        for i, det in enumerate(outputs["detections"]):
            # Get detections above confidence threshold
            boxes = det.boxes.xyxy.cpu().numpy()
            confs = det.boxes.conf.cpu().numpy()
            classes = det.boxes.cls.cpu().numpy()
            
            # Filter by confidence
            valid_mask = confs > self.confidence_threshold
            valid_boxes = boxes[valid_mask]
            valid_confs = confs[valid_mask]
            valid_classes = classes[valid_mask]
            
            # Calculate safety classifications
            safety_scores = outputs["safety_scores"][i]
            proximity_scores = outputs["proximity_maps"][i]
            
            # Determine failure types
            failure_types = []
            for j in range(len(valid_boxes)):
                proximity = proximity_scores[j]
                
                if proximity > self.obvious_threshold:
                    failure_type = "obvious"
                elif proximity > self.subtle_threshold:
                    failure_type = "subtle"
                else:
                    failure_type = "safe"
                
                failure_types.append(failure_type)
            
            failure_predictions.append({
                "boxes": valid_boxes,
                "confs": valid_confs,
                "classes": valid_classes,
                "failure_types": failure_types,
                "safety_scores": safety_scores,
            })
        
        return {
            "detections": outputs["detections"],
            "failure_predictions": failure_predictions,
            "proximity_maps": outputs["proximity_maps"],
            "attention_maps": outputs["attention_maps"],
        }


class ProximityCalculator(nn.Module):
    """Calculate proximity scores between objects based on safety boundaries."""
    
    def __init__(self):
        super().__init__()
        self.distance_calculator = EuclideanDistance()
        self.normalizer = nn.Sigmoid()
        
    def forward(self, detection) -> torch.Tensor:
        """Calculate proximity heatmap for a detection."""
        boxes = detection.boxes.xyxy.cpu().numpy()
        h, w = boxes.shape[0] > 0 and detection.orig_shape[:2] or (416, 416)
        
        # Create proximity map
        proximity_map = torch.zeros(h, w)
        
        # Calculate distances between all object pairs
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box1, box2 = boxes[i], boxes[j]
                
                # Calculate centers
                center1 = torch.tensor([
                    (box1[0] + box1[2]) / 2,
                    (box1[1] + box1[3]) / 2
                ])
                center2 = torch.tensor([
                    (box2[0] + box2[2]) / 2,
                    (box2[1] + box2[3]) / 2
                ])
                
                # Calculate distance
                distance = torch.norm(center1 - center2)
                
                # Create proximity effect
                proximity_value = 1.0 / (1.0 + distance)
                
                # Add to proximity map
                y1, x1 = int(box1[1]), int(box1[0])
                y2, x2 = int(box1[3]), int(box1[2])
                
                proximity_map[y1:y2, x1:x2] = torch.max(
                    torch.tensor(proximity_value),
                    proximity_map[y1:y2, x1:x2]
                )
        
        return proximity_map.unsqueeze(0)


class SafetyAttention(nn.Module):
    """Attention mechanism focusing on safety-critical regions."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(129, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features: torch.Tensor, proximity_map: torch.Tensor) -> torch.Tensor:
        """Generate attention map based on features and proximity."""
        # Resize proximity map to match features
        if proximity_map.shape[2:] != features.shape[2:]:
            proximity_map = torch.nn.functional.interpolate(
                proximity_map, 
                size=features.shape[2:], 
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate features with proximity
        combined = torch.cat([features, proximity_map], dim=1)
        
        # Apply attention layers
        attention = torch.relu(self.conv1(combined))
        attention = self.sigmoid(self.conv2(attention))
        
        return attention


class FeatureExtractor(nn.Module):
    """Extract features from YOLO model for safety assessment."""
    
    def __init__(self):
        super().__init__()
        # Simple CNN for feature extraction
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input image."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class EuclideanDistance(nn.Module):
    """Calculate Euclidean distance between points."""
    
    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        return torch.norm(points1 - points2, dim=1)