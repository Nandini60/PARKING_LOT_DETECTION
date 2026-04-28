# -*- coding: utf-8 -*-
"""Deep dive: WHY is the custom model misclassifying? 
Check what class names the model outputs and their distribution."""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ultralytics import YOLO
import cv2

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'parking_best.pt')
model = YOLO(model_path)

print(f"Model classes: {model.names}")
print(f"Model task: {model.task}")

# Run on test image at different conf levels
img = cv2.imread('test_parking.jpg')
print(f"\nImage shape: {img.shape}")

for conf in [0.5, 0.4, 0.3, 0.25]:
    results = model(img, conf=conf, iou=0.5, imgsz=640, verbose=False)
    empty = 0
    occupied = 0
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cid = int(box.cls[0])
            name = model.names[cid]
            c = float(box.conf[0])
            if 'empty' in name.lower():
                empty += 1
            else:
                occupied += 1
    print(f"\n  conf={conf}: {empty} empty, {occupied} occupied (total={empty+occupied})")

# Also test with generic model
print("\n\n--- GENERIC YOLO (yolov8m) ---")
generic = YOLO('yolov8m.pt')
vehicle_ids = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

for conf in [0.25, 0.15, 0.10, 0.05]:
    results = generic(img, conf=conf, iou=0.5, imgsz=640, verbose=False)
    count = 0
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            if int(box.cls[0]) in vehicle_ids:
                count += 1
    print(f"  conf={conf}: {count} vehicles")

# Now test generic with UPSCALED image
print("\n--- GENERIC YOLO on 2x UPSCALED ---")
upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
for conf in [0.25, 0.15, 0.10]:
    results = generic(upscaled, conf=conf, iou=0.5, imgsz=1280, verbose=False)
    count = 0
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            if int(box.cls[0]) in vehicle_ids:
                count += 1
    print(f"  conf={conf}: {count} vehicles")

# Test generic on tiles
print("\n--- GENERIC YOLO on TILES ---")
h, w = img.shape[:2]
tile_h, tile_w = h//2, w//2
total_from_tiles = set()
for conf in [0.15]:
    for yi in range(2):
        for xi in range(2):
            y1, x1 = yi * (h//2), xi * (w//2)
            y2, x2 = min(y1 + tile_h + 100, h), min(x1 + tile_w + 100, w)
            tile = img[y1:y2, x1:x2]
            results = generic(tile, conf=conf, iou=0.5, imgsz=640, verbose=False)
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    if int(box.cls[0]) in vehicle_ids:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                        cx, cy = int((bx1+bx2)/2)+x1, int((by1+by2)/2)+y1
                        total_from_tiles.add((cx//20, cy//20))  # grid-snap to avoid duplicates
    print(f"  conf={conf}: ~{len(total_from_tiles)} unique vehicles from tiles")

# Test custom model on tiles  
print("\n--- CUSTOM MODEL on TILES ---")
total_custom = {'empty': 0, 'occupied': 0}
for yi in range(2):
    for xi in range(2):
        y1, x1 = yi * (h//2), xi * (w//2)
        y2, x2 = min(y1 + tile_h + 100, h), min(x1 + tile_w + 100, w)
        tile = img[y1:y2, x1:x2]
        results = model(tile, conf=0.3, iou=0.5, imgsz=640, verbose=False)
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                name = model.names[int(box.cls[0])]
                if 'empty' in name.lower():
                    total_custom['empty'] += 1
                else:
                    total_custom['occupied'] += 1
print(f"  Custom tiles: {total_custom}")
