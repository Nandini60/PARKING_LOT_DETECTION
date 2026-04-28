# -*- coding: utf-8 -*-
"""Diagnose which model is running and test accuracy."""
import os, sys, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 60)
print("  ParkVision AI - Model Diagnostic Report")
print("=" * 60)

# 1. Check model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'parking_best.pt')
print(f"\n[1] Model file: {model_path}")
print(f"    Exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    size = os.path.getsize(model_path) / (1024*1024)
    mtime = time.ctime(os.path.getmtime(model_path))
    print(f"    Size: {size:.1f} MB")
    print(f"    Last modified: {mtime}")

# 2. Check training runs
v1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_runs', 'parkvision')
v2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_runs', 'parkvision_v2')
print(f"\n[2] Training runs:")
print(f"    v1 (parkvision): {'EXISTS' if os.path.exists(v1_dir) else 'NOT FOUND'}")
print(f"    v2 (parkvision_v2): {'EXISTS' if os.path.exists(v2_dir) else 'NOT FOUND'}")

# Check which best.pt is newer
v1_best = os.path.join(v1_dir, 'weights', 'best.pt')
v2_best = os.path.join(v2_dir, 'weights', 'best.pt')
if os.path.exists(v1_best):
    print(f"    v1 best.pt modified: {time.ctime(os.path.getmtime(v1_best))}")
if os.path.exists(v2_best):
    print(f"    v2 best.pt modified: {time.ctime(os.path.getmtime(v2_best))}")

# Compare file sizes to see which one is deployed
if os.path.exists(model_path) and os.path.exists(v2_best):
    deployed_size = os.path.getsize(model_path)
    v2_size = os.path.getsize(v2_best)
    print(f"\n    Deployed model size: {deployed_size}")
    print(f"    v2 best.pt size: {v2_size}")
    print(f"    MATCH: {'YES - v2 model is deployed' if deployed_size == v2_size else 'NO - different model!'}")

# 3. Load and inspect the model
print(f"\n[3] Loading model...")
from ultralytics import YOLO
model = YOLO(model_path)
print(f"    Classes: {model.names}")
print(f"    Number of classes: {len(model.names)}")
print(f"    Model type: {model.task}")

# 4. Check dataset sizes
ds_v1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'parking')
ds_v2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'parking_v2')
print(f"\n[4] Datasets:")
for name, path in [("v1 (parking)", ds_v1), ("v2 (parking_v2)", ds_v2)]:
    train_imgs = os.path.join(path, 'train', 'images')
    val_imgs = os.path.join(path, 'valid', 'images')
    if os.path.exists(train_imgs):
        train_count = len([f for f in os.listdir(train_imgs) if f.endswith(('.jpg','.png'))])
        val_count = len([f for f in os.listdir(val_imgs) if f.endswith(('.jpg','.png'))]) if os.path.exists(val_imgs) else 0
        print(f"    {name}: {train_count} train, {val_count} val")
    else:
        print(f"    {name}: NOT FOUND")

# 5. Quick accuracy test
print(f"\n[5] Running detection on test image...")
from parking_detector import ParkingDetector
det = ParkingDetector()
print(f"    Custom mode: {det.custom_mode}")
print(f"    Confidence threshold: {det.conf}")

test_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_parking.jpg')
if os.path.exists(test_img):
    result = det.analyze(test_img, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_output'))
    print(f"    Total: {result['total_spaces']}")
    print(f"    Occupied: {result['occupied']}")
    print(f"    Available: {result['available']}")
    print(f"    Avg confidence: {result['confidence_avg']}%")
    print(f"    Model name: {result['model_info']['name']}")
    
    # Show confidence distribution
    confs = [d['confidence'] for d in result['detections']]
    if confs:
        high = sum(1 for c in confs if c >= 0.7)
        med = sum(1 for c in confs if 0.4 <= c < 0.7)
        low = sum(1 for c in confs if c < 0.4)
        print(f"\n    Confidence breakdown:")
        print(f"      High (>=70%): {high} detections")
        print(f"      Medium (40-70%): {med} detections")
        print(f"      Low (<40%): {low} detections")

print(f"\n[6] ROOT CAUSE ANALYSIS:")
print(f"    The model was trained on SYNTHETIC data (computer-generated images).")
print(f"    Synthetic parking lots look NOTHING like real aerial photos.")
print(f"    This causes a 'domain gap' - high accuracy on synthetic data,")
print(f"    but lower accuracy on real-world images.")
print(f"\n    SOLUTION: Use a HYBRID approach combining:")
print(f"    - Generic YOLO (trained on millions of real images) for vehicle detection")
print(f"    - Custom model for parking space detection")
print(f"    - Merge both results for maximum accuracy")
print(f"\n" + "=" * 60)
