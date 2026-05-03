import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


@dataclass
class SafetyAnnotation:
    """Safety boundary annotations for objects"""
    object_id: int
    category: str
    bbox: List[float]  # [x, y, w, h]
    safety_zone: float  # meters
    boundary_type: str
    critical_objects: List[str]
    is_failure: bool  # True if boundary violated
    failure_type: str  # "subtle" or "obvious"
    proximity_score: float  # 0-1, how close to boundary


@dataclass
class SafetySample:
    """Sample with safety annotations"""
    image_path: str
    image_id: int
    annotations: List[SafetyAnnotation]
    failure_mask: np.ndarray  # Binary mask showing failures
    safety_heatmap: np.ndarray  # Heatmap of safety zones
    metadata: Dict


class SafetyBoundaryDataset:
    """
    COCO dataset adapted for safety boundary detection.
    
    Creates subtle vs obvious failure scenarios based on safety boundary definitions.
    """
    
    def __init__(
        self,
        coco_path: str,
        safety_boundaries: Dict,
        split: str = "train",
        image_size: Tuple[int, int] = (640, 640),
        subtle_threshold: float = 0.3,
        obvious_threshold: float = 0.8,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.coco_path = Path(coco_path)
        self.safety_boundaries = safety_boundaries
        self.split = split
        self.image_size = image_size
        self.subtle_threshold = subtle_threshold
        self.obvious_threshold = obvious_threshold
        self.augment = augment
        self.max_samples = max_samples
        
        # Initialize COCO API (detect available annotation file)
        if (self.coco_path / "annotations" / "instances_val2017.json").exists():
            ann_path = self.coco_path / "annotations" / "instances_val2017.json"
        elif (self.coco_path / "instances_val2017.json").exists():
            ann_path = self.coco_path / "instances_val2017.json"
        else:
            raise FileNotFoundError(f"COCO annotation file not found in {self.coco_path}")
        self.coco = COCO(str(ann_path))
        
        # Load image information
        self.image_ids = list(self.coco.imgs.keys())
        if split == "train":
            self.image_ids = self.image_ids[:int(0.7 * len(self.image_ids))]
        elif split == "val":
            self.image_ids = self.image_ids[int(0.7 * len(self.image_ids)):int(0.85 * len(self.image_ids))]
        else:  # test
            self.image_ids = self.image_ids[int(0.85 * len(self.image_ids))::]
        
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]
        
        # Define augmentations
        if augment:
            self.transform = A.Compose([
                A.Resize(*image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        # Safety boundary categories
        self.safety_categories = {
            1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
            6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
            # Add more COCO categories as needed
        }
        
        print(f"Initialized {split} split with {len(self.image_ids)} images")
    
    def _get_safety_boundary(self, category_name: str) -> Optional[Dict]:
        """Get safety boundary definition for a category"""
        return self.safety_boundaries.get(category_name)
    
    def _calculate_proximity_score(
        self, 
        obj_bbox: List[float], 
        critical_obj_bbox: List[float],
        safety_zone: float
    ) -> float:
        """Calculate proximity score between two objects. Returns 0-1 where 1 means very close."""
        if not obj_bbox or not critical_obj_bbox:
            return 0.0
        
        # Calculate center points
        obj_center = np.array([obj_bbox[0] + obj_bbox[2]/2, obj_bbox[1] + obj_bbox[3]/2])
        crit_center = np.array([critical_obj_bbox[0] + critical_obj_bbox[2]/2, critical_obj_bbox[1] + critical_obj_bbox[3]/2])
        
        # Calculate pixel distance
        pixel_distance = np.linalg.norm(obj_center - crit_center)
        
        # Convert to proximity score: closer objects have higher score
        # Scale factor: assume safety_zone represents ~100 pixels worth of safe distance
        scale_factor = safety_zone * 100.0
        if scale_factor > 0:
            proximity = 1.0 - min(1.0, pixel_distance / scale_factor)
            proximity = max(0.0, proximity)
        else:
            proximity = 0.0
        
        return proximity
    
    def _generate_failure_scenarios(self, image_info: Dict) -> List[SafetyAnnotation]:
        """
        Generate subtle and obvious failure scenarios based on safety boundaries.
        """
        annotations = []
        ann_ids = self.coco.getAnnIds(imgIds=image_info["id"])
        anns = self.coco.loadAnns(ann_ids)
        
        # Group annotations by category
        category_anns = {}
        for ann in anns:
            cat_name = self.safety_categories.get(ann["category_id"], f"unknown_{ann['category_id']}")
            if cat_name not in category_anns:
                category_anns[cat_name] = []
            category_anns[cat_name].append(ann)
        
        # Generate safety annotations
        for cat_name, cat_anns in category_anns.items():
            safety_info = self._get_safety_boundary(cat_name)
            if not safety_info:
                continue
            
            for ann in cat_anns:
                obj_bbox = ann["bbox"]  # [x, y, w, h]
                
                # Find critical objects in proximity
                proximity_score = 0.0
                is_failure = False
                failure_type = "success"
                
                for other_cat, other_anns in category_anns.items():
                    if other_cat in safety_info["critical_objects"]:
                        for other_ann in other_anns:
                            critical_bbox = other_ann["bbox"]
                            proximity_score = self._calculate_proximity_score(
                                obj_bbox, critical_bbox, safety_info["safety_zone"]
                            )
                            
                            # Classify failure type based on proximity
                            if proximity_score >= self.obvious_threshold:
                                is_failure = True
                                failure_type = "obvious"
                            elif proximity_score >= self.subtle_threshold:
                                is_failure = True
                                failure_type = "subtle"
                            
                            break
                
                safety_annotation = SafetyAnnotation(
                    object_id=ann["id"],
                    category=cat_name,
                    bbox=obj_bbox,
                    safety_zone=safety_info["safety_zone"],
                    boundary_type=safety_info["boundary_type"],
                    critical_objects=safety_info["critical_objects"],
                    is_failure=is_failure,
                    failure_type=failure_type,
                    proximity_score=proximity_score,
                )
                annotations.append(safety_annotation)
        
        return annotations
    
    def _create_safety_heatmap(self, annotations: List[SafetyAnnotation], image_size: Tuple[int, int]) -> np.ndarray:
        """Create safety heatmap showing boundary zones"""
        heatmap = np.zeros(image_size)
        
        for ann in annotations:
            if ann.is_failure:
                # Create gradient effect around boundary
                x, y, w, h = ann.bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # Add failure intensity to heatmap
                intensity = ann.proximity_score
                heatmap[y1:y2, x1:x2] = max(heatmap[y1:y2, x1:x2], intensity)
        
        return heatmap
    
    def _create_failure_mask(self, annotations: List[SafetyAnnotation], image_size: Tuple[int, int]) -> np.ndarray:
        """Create binary mask showing failure regions"""
        mask = np.zeros(image_size, dtype=np.uint8)
        
        for ann in annotations:
            if ann.is_failure:
                x, y, w, h = ann.bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample with safety annotations"""
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        
        # Load image
        image_path = self.coco_path / "val2017" / image_info["file_name"]
        if not image_path.is_file():
            # Create a dummy image (zeros) with configured size
            dummy = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            image = dummy
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate safety annotations
        annotations = self._generate_failure_scenarios(image_info)
        
        # Create safety heatmap and failure mask
        safety_heatmap = self._create_safety_heatmap(annotations, image.shape[:2])
        failure_mask = self._create_failure_mask(annotations, image.shape[:2])
        
        # Apply augmentations
        transformed = self.transform(image=image, mask=failure_mask, heatmap=safety_heatmap)
        
        # Prepare sample
        sample = SafetySample(
            image_path=str(image_path),
            image_id=image_id,
            annotations=annotations,
            failure_mask=transformed["mask"],
            safety_heatmap=transformed["heatmap"],
            metadata={
                "width": image_info["width"],
                "height": image_info["height"],
                "file_name": image_info["file_name"],
                "num_failures": sum(1 for ann in annotations if ann.is_failure),
                "num_subtle": sum(1 for ann in annotations if ann.failure_type == "subtle"),
                "num_obvious": sum(1 for ann in annotations if ann.failure_type == "obvious"),
            }
        )
        
        # Determine majority safety label for the image (0=safe, 1=subtle, 2=obvious)
        label_counts = {0: 0, 1: 0, 2: 0}
        for ann in annotations:
            if ann.failure_type == "obvious":
                label_counts[2] += 1
            elif ann.failure_type == "subtle":
                label_counts[1] += 1
            else:
                label_counts[0] += 1
        majority_label = max(label_counts, key=label_counts.get)
        return {
            "image": transformed["image"],
            "sample": sample,
            "failure_mask": transformed["mask"].float(),
            "safety_heatmap": torch.from_numpy(transformed["heatmap"]).float(),
            "label": torch.tensor(majority_label, dtype=torch.long),
        }
    
    def get_failure_statistics(self) -> Dict:
        """Get statistics about failure distribution"""
        total_failures = 0
        total_subtle = 0
        total_obvious = 0
        total_images = 0
        
        for idx in range(min(100, len(self))):  # Sample first 100
            sample = self[idx]
            stats = sample["sample"].metadata
            
            total_failures += stats["num_failures"]
            total_subtle += stats["num_subtle"]
            total_obvious += stats["num_obvious"]
            total_images += 1
        
        return {
            "total_images": total_images,
            "total_failures": total_failures,
            "subtle_failures": total_subtle,
            "obvious_failures": total_obvious,
            "subtle_ratio": total_subtle / max(total_failures, 1),
            "obvious_ratio": total_obvious / max(total_failures, 1),
        }


def create_dataloader(
    dataset_path: str,
    safety_boundaries: Dict,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Create data loader for safety boundary detection"""
    dataset = SafetyBoundaryDataset(
        coco_path=dataset_path,
        safety_boundaries=safety_boundaries,
        split=split,
        augment=(split == "train"),
        max_samples=max_samples,
    )
    
    # Custom collate: stack tensor fields, keep others as list
    def collate_fn(batch):
        batch_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                batch_dict[key] = torch.stack([item[key] for item in batch])
            else:
                batch_dict[key] = [item[key] for item in batch]
        return batch_dict
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    
    return dataloader, dataset