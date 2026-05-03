import os
os.chdir(r"D:\Professors Reached Tucker\Notebooks")

from pathlib import Path
from ultralytics import YOLO
import shutil

COCO_PATH = Path("val2017")
SAFE_PATH = Path("Safe")
SUBTLE_PATH = Path("Subtle")
OBVIOUS_PATH = Path("Obvious")

# Categories
SAFE_CATS = {'person', 'chair', 'cup', 'bowl', 'apple', 'banana', 'book', 'clock', 'tv', 'laptop', 'cell phone', 'potted plant', 'bed', 'couch', 'dining table', 'train', 'truck', 'bus', 'car', 'bird', 'cat', 'dog', 'horse'}
SUBTLE_CATS = {'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'snowboard', 'kite', 'baseball bat', 'skateboard', 'surfboard', 'tennis racket'}
OBVIOUS_CATS = {'knife', 'scissors', 'gun'}

model = YOLO('yolov8n.pt')

# Clear folders
for p in [SAFE_PATH, SUBTLE_PATH, OBVIOUS_PATH]:
    for f in p.glob('*.jpg'): f.unlink()

safe_count = subtle_count = obvious_count = 0

for img in sorted(COCO_PATH.glob('*.jpg'))[:1500]:  # Process 1500 images
    results = model(img, verbose=False)
    cats = set()
    for r in results:
        if r.boxes is not None:
            for b in r.boxes:
                cats.add(model.names[int(b.cls[0])])
    
    if cats & OBVIOUS_CATS and obvious_count < 300:
        shutil.copy(img, OBVIOUS_PATH / f"obvious_{obvious_count:04d}.jpg")
        obvious_count += 1
    elif cats & SUBTLE_CATS and subtle_count < 200:
        shutil.copy(img, SUBTLE_PATH / f"subtle_{subtle_count:04d}.jpg")
        subtle_count += 1
    elif cats and safe_count < 100:
        shutil.copy(img, SAFE_PATH / f"safe_{safe_count:04d}.jpg")
        safe_count += 1

print(f"Safe: {safe_count}, Subtle: {subtle_count}, Obvious: {obvious_count}")