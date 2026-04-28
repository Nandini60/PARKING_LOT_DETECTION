# -*- coding: utf-8 -*-
"""Calibrate texture thresholds by analyzing actual detections."""
import sys, os, cv2, numpy as np
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ultralytics import YOLO

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'parking_best.pt')
model = YOLO(model_path)

img = cv2.imread('test_parking.jpg')
h, w = img.shape[:2]

results = model(img, conf=0.40, iou=0.45, imgsz=640, verbose=False)

print(f"{'Type':<12} {'Conf':>5} {'Edge':>7} {'Color':>7} {'Grad':>7} {'TOTAL':>7} {'Verdict'}")
print("-" * 65)

for r in results:
    if r.boxes is None: continue
    for box in r.boxes:
        cid = int(box.cls[0])
        name = model.names[cid]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c - x1c < 10 or y2c - y1c < 10:
            continue
        
        roi = img[y1c:y2c, x1c:x2c]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Edge score
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        edge = np.mean(np.abs(lap))
        
        # Color variance
        color_std = np.mean([np.std(roi[:,:,c]) for c in range(3)])
        
        # Gradient
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.mean(np.sqrt(gx**2 + gy**2))
        
        total_score = edge * 0.4 + color_std * 0.3 + grad * 0.3
        
        space_type = 'empty' if 'empty' in name.lower() else 'occupied'
        verdict = 'CAR' if total_score >= 18 else 'EMPTY'
        
        print(f"{space_type:<12} {conf:>5.2f} {edge:>7.1f} {color_std:>7.1f} {grad:>7.1f} {total_score:>7.1f}  {verdict}")
