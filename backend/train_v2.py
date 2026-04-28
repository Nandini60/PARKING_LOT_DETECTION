# -*- coding: utf-8 -*-
"""
ParkVision AI - Advanced Dataset Generator + Model Trainer v2
Generates 1500+ highly realistic parking lot training images and trains YOLOv8.
"""

import os, sys, math, time, shutil, random
import numpy as np
import cv2

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Patch matplotlib for Windows DLL issue
import types
try:
    import matplotlib._image
except (ImportError, OSError):
    fake = types.ModuleType('matplotlib._image')
    fake.resample = lambda *a, **k: None
    sys.modules['matplotlib._image'] = fake
try:
    import ultralytics.utils.checks as uc
    orig = uc.check_font
    uc.check_font = lambda *a, **k: None
except: pass
os.environ['MPLBACKEND'] = 'Agg'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "parking_v2")

# ====== Realistic car colors (BGR) ======
CAR_COLORS = [
    (40, 40, 40), (50, 50, 50), (60, 60, 70),       # Black/dark gray
    (200, 200, 200), (220, 220, 220), (240, 240, 240),# White/silver
    (180, 180, 190), (160, 165, 170),                  # Silver
    (50, 50, 180), (30, 30, 200), (40, 40, 160),      # Red
    (180, 80, 40), (200, 100, 50),                      # Blue
    (80, 120, 50), (60, 100, 40),                       # Green
    (30, 140, 200), (20, 120, 180),                     # Orange/yellow
    (100, 100, 120), (80, 80, 100),                     # Dark gray
    (60, 80, 140), (50, 60, 120),                       # Brown/maroon
]

def draw_realistic_car(img, x, y, w, h, angle=0):
    """Draw a more realistic-looking car from top-down view."""
    color = random.choice(CAR_COLORS)
    # Add slight color variation
    color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in color)
    
    cx, cy = x + w//2, y + h//2
    
    # Car body (slightly rounded rectangle)
    body_pts = np.array([
        [x + w*0.1, y + h*0.05],
        [x + w*0.9, y + h*0.05],
        [x + w*0.95, y + h*0.15],
        [x + w*0.95, y + h*0.85],
        [x + w*0.9, y + h*0.95],
        [x + w*0.1, y + h*0.95],
        [x + w*0.05, y + h*0.85],
        [x + w*0.05, y + h*0.15],
    ], dtype=np.int32)
    
    if angle != 0:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        body_pts = np.array([cv2.transform(body_pts.reshape(1, -1, 2).astype(np.float32), M)[0]], dtype=np.int32)[0]
    
    cv2.fillPoly(img, [body_pts], color)
    
    # Windshield (darker)
    ws_color = tuple(max(0, c - 50) for c in color)
    ws_h = int(h * 0.25)
    ws_pts = np.array([
        [x + w*0.15, y + h*0.08],
        [x + w*0.85, y + h*0.08],
        [x + w*0.8, y + h*0.08 + ws_h],
        [x + w*0.2, y + h*0.08 + ws_h],
    ], dtype=np.int32)
    if angle != 0:
        ws_pts = np.array([cv2.transform(ws_pts.reshape(1, -1, 2).astype(np.float32), M)[0]], dtype=np.int32)[0]
    cv2.fillPoly(img, [ws_pts], ws_color)
    
    # Rear window
    rw_pts = np.array([
        [x + w*0.2, y + h*0.7],
        [x + w*0.8, y + h*0.7],
        [x + w*0.85, y + h*0.88],
        [x + w*0.15, y + h*0.88],
    ], dtype=np.int32)
    if angle != 0:
        rw_pts = np.array([cv2.transform(rw_pts.reshape(1, -1, 2).astype(np.float32), M)[0]], dtype=np.int32)[0]
    cv2.fillPoly(img, [rw_pts], ws_color)
    
    # Roof highlight
    roof_color = tuple(min(255, c + 25) for c in color)
    roof_pts = np.array([
        [x + w*0.25, y + h*0.3],
        [x + w*0.75, y + h*0.3],
        [x + w*0.7, y + h*0.65],
        [x + w*0.3, y + h*0.65],
    ], dtype=np.int32)
    if angle != 0:
        roof_pts = np.array([cv2.transform(roof_pts.reshape(1, -1, 2).astype(np.float32), M)[0]], dtype=np.int32)[0]
    cv2.fillPoly(img, [roof_pts], roof_color)
    
    # Shadow under car
    shadow = img.copy()
    shadow_pts = np.array([
        [x - 3, y + int(h*0.9)],
        [x + w + 3, y + int(h*0.9)],
        [x + w + 5, y + h + 4],
        [x - 5, y + h + 4],
    ], dtype=np.int32)
    if angle != 0:
        shadow_pts = np.array([cv2.transform(shadow_pts.reshape(1, -1, 2).astype(np.float32), M)[0]], dtype=np.int32)[0]
    cv2.fillPoly(shadow, [shadow_pts], (0, 0, 0))
    cv2.addWeighted(shadow, 0.15, img, 0.85, 0, img)


def generate_parking_lot_image(img_size=640):
    """Generate a realistic parking lot image with labeled spaces."""
    # Asphalt background with realistic texture
    base_gray = random.randint(80, 140)
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * base_gray
    
    # Add asphalt texture
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add subtle patches/stains on asphalt
    for _ in range(random.randint(3, 10)):
        cx = random.randint(0, img_size)
        cy = random.randint(0, img_size)
        r = random.randint(20, 80)
        stain_val = random.randint(-20, 20)
        cv2.circle(img, (cx, cy), r, tuple(max(0, min(255, base_gray + stain_val)) for _ in range(3)), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    labels = []
    layout = random.choice(['perpendicular', 'angled', 'parallel', 'mixed'])
    
    # Generate parking rows
    num_rows = random.randint(2, 5)
    margin = random.randint(20, 50)
    row_gap = random.randint(60, 100)  # driving lane between rows
    
    y_cursor = margin
    
    for row in range(num_rows):
        if y_cursor + 80 > img_size - margin:
            break
            
        angle = 0
        if layout == 'angled':
            angle = random.choice([-30, -45, -60, 30, 45, 60])
        elif layout == 'mixed':
            angle = random.choice([0, 0, 0, -45, 45])
        
        # Determine space dimensions based on angle
        if abs(angle) > 20:
            space_w = random.randint(35, 55)
            space_h = random.randint(65, 95)
        elif layout == 'parallel':
            space_w = random.randint(70, 100)
            space_h = random.randint(30, 45)
        else:
            space_w = random.randint(40, 65)
            space_h = random.randint(70, 100)
        
        num_cols = min(random.randint(4, 12), (img_size - 2 * margin) // (space_w + 4))
        
        # Center the row
        row_width = num_cols * (space_w + 4)
        x_start = (img_size - row_width) // 2
        
        # Draw driving lane between rows
        if row > 0:
            lane_y = y_cursor - row_gap // 2
            # Lane markings (dashed center line)
            for lx in range(margin, img_size - margin, 30):
                cv2.line(img, (lx, lane_y), (min(lx + 15, img_size - margin), lane_y), 
                        (base_gray + 40, base_gray + 40, base_gray + 30), 1)
        
        for col in range(num_cols):
            sx = x_start + col * (space_w + 4)
            sy = y_cursor
            
            if sx + space_w > img_size - margin or sy + space_h > img_size - margin:
                continue
            
            # Draw parking space lines (white/yellow)
            line_color = random.choice([(200, 200, 200), (210, 210, 210), (180, 190, 200), (180, 200, 220)])
            line_thickness = random.choice([1, 2])
            
            # Left line
            cv2.line(img, (sx, sy), (sx, sy + space_h), line_color, line_thickness)
            # Right line
            cv2.line(img, (sx + space_w, sy), (sx + space_w, sy + space_h), line_color, line_thickness)
            # Bottom line (sometimes)
            if random.random() > 0.3:
                cv2.line(img, (sx, sy + space_h), (sx + space_w, sy + space_h), line_color, line_thickness)
            
            # Decide if occupied (60-75% occupancy rate)
            occupied = random.random() < random.uniform(0.55, 0.80)
            
            if occupied:
                # Draw car with slight offset (not perfectly centered)
                car_margin_x = random.randint(3, max(4, space_w // 8))
                car_margin_y = random.randint(3, max(4, space_h // 8))
                car_offset_x = random.randint(-3, 3)
                car_offset_y = random.randint(-3, 3)
                
                car_x = sx + car_margin_x + car_offset_x
                car_y = sy + car_margin_y + car_offset_y
                car_w = space_w - 2 * car_margin_x
                car_h = space_h - 2 * car_margin_y
                
                if car_w > 15 and car_h > 15:
                    car_angle = random.uniform(-5, 5)  # Slight imperfect parking
                    draw_realistic_car(img, car_x, car_y, car_w, car_h, car_angle)
                
                cls_id = 1  # space-occupied
            else:
                cls_id = 0  # space-empty
                # Sometimes add tire marks or oil stains in empty spaces
                if random.random() > 0.6:
                    stain_x = sx + space_w // 2 + random.randint(-10, 10)
                    stain_y = sy + space_h // 2 + random.randint(-10, 10)
                    stain_r = random.randint(5, 15)
                    stain_color = tuple(max(0, base_gray - random.randint(10, 30)) for _ in range(3))
                    cv2.circle(img, (stain_x, stain_y), stain_r, stain_color, -1)
            
            # YOLO format label
            cx_norm = (sx + space_w / 2) / img_size
            cy_norm = (sy + space_h / 2) / img_size
            w_norm = space_w / img_size
            h_norm = space_h / img_size
            labels.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        y_cursor += space_h + row_gap
    
    # Add global lighting variation
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    
    # Add shadows (simulate time of day)
    if random.random() > 0.5:
        shadow_overlay = img.copy()
        shadow_angle = random.uniform(0, math.pi)
        shadow_x = int(img_size * 0.3 * math.cos(shadow_angle))
        shadow_y = int(img_size * 0.3 * math.sin(shadow_angle))
        pts = np.array([
            [0, 0], [img_size // 2 + shadow_x, shadow_y],
            [img_size // 2 + shadow_x + 100, img_size],
            [0, img_size]
        ], dtype=np.int32)
        cv2.fillPoly(shadow_overlay, [pts], (0, 0, 0))
        cv2.addWeighted(shadow_overlay, 0.15, img, 0.85, 0, img)
    
    # Random hue shift (simulate different lighting conditions)
    if random.random() > 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return img, labels


def generate_dataset():
    """Generate large realistic dataset."""
    print("[1/3] Generating realistic parking lot dataset...")
    
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)
    
    # 1200 training + 300 validation images
    splits = [("train", 1200), ("valid", 300)]
    
    for split, count in splits:
        print(f"   Generating {count} {split} images...")
        for i in range(count):
            # Vary image size for more robustness
            img_size = random.choice([416, 480, 512, 576, 640])
            img, labels = generate_parking_lot_image(img_size)
            
            # Resize all to 640 for consistency
            if img_size != 640:
                img = cv2.resize(img, (640, 640))
            
            img_path = os.path.join(DATASET_DIR, split, "images", f"lot_{i:05d}.jpg")
            lbl_path = os.path.join(DATASET_DIR, split, "labels", f"lot_{i:05d}.txt")
            
            # Random JPEG quality for robustness
            quality = random.randint(70, 95)
            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(labels))
            
            if (i + 1) % 200 == 0:
                print(f"      {i+1}/{count} done")
    
    # Write data.yaml
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.join(DATASET_DIR, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(DATASET_DIR, 'valid', 'images')}\n")
        f.write("nc: 2\n")
        f.write("names: ['space-empty', 'space-occupied']\n")
    
    print(f"   [OK] Dataset: 1200 train + 300 val images")
    print(f"   [OK] data.yaml: {yaml_path}")
    return yaml_path


def train_model(data_yaml, epochs=50, batch=16, imgsz=416):
    """Train YOLOv8 on the dataset."""
    print(f"\n[2/3] Training YOLOv8n model ({epochs} epochs, batch={batch}, imgsz={imgsz})...")
    
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='parkvision_v2',
        project=os.path.join(BASE_DIR, 'training_runs'),
        exist_ok=True,
        patience=15,
        save=True,
        plots=False,
        verbose=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    
    return results


def save_model():
    """Copy best model to models folder."""
    print("\n[3/3] Saving trained model...")
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    best_pt = os.path.join(BASE_DIR, 'training_runs', 'parkvision_v2', 'weights', 'best.pt')
    target_pt = os.path.join(models_dir, 'parking_best.pt')
    
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, target_pt)
        size_mb = os.path.getsize(target_pt) / (1024 * 1024)
        print(f"   [OK] Model saved: {target_pt} ({size_mb:.1f} MB)")
        return True
    else:
        last_pt = best_pt.replace('best.pt', 'last.pt')
        if os.path.exists(last_pt):
            shutil.copy2(last_pt, target_pt)
            print(f"   [OK] Model saved: {target_pt}")
            return True
    
    print("   [FAIL] No weights found!")
    return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  ParkVision AI - Advanced Model Training v2")
    print("  1200 train + 300 val realistic images | 50 epochs")
    print("=" * 60 + "\n")
    
    t0 = time.time()
    
    # Step 1: Generate dataset
    data_yaml = generate_dataset()
    
    # Step 2: Train
    train_model(data_yaml, epochs=50, batch=16, imgsz=416)
    
    # Step 3: Save
    if save_model():
        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"  TRAINING COMPLETE in {elapsed/60:.1f} minutes!")
        print(f"  Restart server to use the new model.")
        print(f"{'=' * 60}\n")
    else:
        print("\n  Training failed!")
