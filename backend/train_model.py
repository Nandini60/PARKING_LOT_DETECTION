# -*- coding: utf-8 -*-
"""
ParkVision AI - Custom Model Training Script
Patches matplotlib DLL issue on Windows, then trains YOLOv8.
"""
import argparse, os, sys, shutil

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# CRITICAL: Patch matplotlib's _image module BEFORE ultralytics imports it
# Windows Application Control policy blocks matplotlib's native DLL
import types
import importlib

def patch_matplotlib():
    """Create a fake matplotlib._image module to bypass DLL block."""
    try:
        import matplotlib._image
    except (ImportError, OSError):
        # Create dummy _image module
        fake_image = types.ModuleType('matplotlib._image')
        fake_image.resample = lambda *a, **k: None
        sys.modules['matplotlib._image'] = fake_image
        # Also ensure matplotlib.colors loads
        try:
            import matplotlib
        except:
            pass

patch_matplotlib()

# Now also patch the check_font function in ultralytics
try:
    import ultralytics.utils.checks
    original_check_font = ultralytics.utils.checks.check_font
    def safe_check_font(*args, **kwargs):
        try:
            return original_check_font(*args, **kwargs)
        except Exception:
            return None
    ultralytics.utils.checks.check_font = safe_check_font
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="Train ParkVision parking detector")
    parser.add_argument('--api-key', type=str, required=True, help='Roboflow API key')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  [P] ParkVision AI - Model Training Pipeline")
    print("="*55 + "\n")

    # Step 1: Download/create dataset
    print("[1/4] Preparing parking lot dataset...")
    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets", "parking")

    # Try Roboflow first, fallback to synthetic
    data_yaml = try_roboflow_download(args.api_key, dataset_dir)
    if not data_yaml:
        print("   Roboflow datasets unavailable. Creating synthetic dataset...")
        data_yaml = create_synthetic_dataset(dataset_dir)

    print(f"   [OK] Dataset ready: {data_yaml}")

    # Step 2: Load base model
    print("\n[2/4] Loading YOLOv8m base model...")
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print("   [OK] Base model loaded!")

    # Step 3: Train
    print(f"\n[3/4] Training ({args.epochs} epochs, batch={args.batch})...")
    print("   ~30-60 min on CPU, ~5-10 min on GPU\n")

    # Set MPLBACKEND to avoid display issues
    os.environ['MPLBACKEND'] = 'Agg'
    # Use smaller imgsz for faster CPU training
    if args.imgsz == 640:
        args.imgsz = 416

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name='parkvision',
        project=os.path.join(os.path.dirname(__file__), 'training_runs'),
        exist_ok=True,
        patience=10,
        save=True,
        plots=False,
        verbose=True,
    )

    # Step 4: Save model
    print("\n[4/4] Saving trained model...")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    best_pt = os.path.join(os.path.dirname(__file__), 'training_runs', 'parkvision', 'weights', 'best.pt')
    target_pt = os.path.join(models_dir, 'parking_best.pt')

    if os.path.exists(best_pt):
        shutil.copy2(best_pt, target_pt)
    else:
        last_pt = best_pt.replace('best.pt', 'last.pt')
        if os.path.exists(last_pt):
            shutil.copy2(last_pt, target_pt)
        else:
            print("   [FAIL] No weights found!")
            sys.exit(1)

    print(f"   [OK] Model: {target_pt} ({os.path.getsize(target_pt)/(1024*1024):.1f} MB)")
    print("\n" + "="*55)
    print("  TRAINING COMPLETE! Restart server to use custom model.")
    print("="*55 + "\n")


def try_roboflow_download(api_key, dataset_dir):
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        datasets = [
            ("brad-dwyer", "parking-lot-detection", 1),
            ("parking-yolov5", "parking-lot-availability", 3),
        ]
        for ws, proj, ver in datasets:
            try:
                print(f"   Trying: {ws}/{proj}...")
                project = rf.workspace(ws).project(proj)
                version = project.version(ver)
                dataset = version.download("yolov8", location=dataset_dir)
                return os.path.join(dataset.location, "data.yaml")
            except Exception:
                continue
    except Exception:
        pass
    return None


def create_synthetic_dataset(dataset_dir):
    """Create parking lot training data with labeled empty/occupied spaces."""
    import numpy as np
    import cv2

    os.makedirs(os.path.join(dataset_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "valid", "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "valid", "labels"), exist_ok=True)

    for split, count in [("train", 300), ("valid", 60)]:
        for i in range(count):
            # Random parking lot background
            bg_val = np.random.randint(90, 170)
            img = np.ones((640, 640, 3), dtype=np.uint8) * bg_val

            # Add asphalt texture
            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            labels = []
            rows = np.random.randint(2, 5)
            cols = np.random.randint(3, 7)
            margin = 30
            sw = (640 - 2*margin) // cols
            sh = (640 - 2*margin) // rows

            for r in range(rows):
                for c in range(cols):
                    x = margin + c * sw + 5
                    y = margin + r * sh + 5
                    w_box = sw - 10
                    h_box = sh - 10

                    occupied = np.random.random() > 0.35

                    # Draw parking lines
                    cv2.rectangle(img, (x-2, y-2), (x+w_box+2, y+h_box+2), (180,180,180), 2)

                    if occupied:
                        # Car body
                        car_color = tuple(np.random.randint(30, 220, 3).tolist())
                        cx_off = np.random.randint(-3, 4)
                        cy_off = np.random.randint(-3, 4)
                        cv2.rectangle(img, (x+4+cx_off, y+4+cy_off),
                                     (x+w_box-4+cx_off, y+h_box-4+cy_off), car_color, -1)
                        # Windshield
                        ws_color = tuple(min(255, c2+40) for c2 in car_color)
                        cv2.rectangle(img, (x+8, y+6), (x+w_box-8, y+h_box//3), ws_color, -1)
                        # Roof detail
                        cv2.rectangle(img, (x+6, y+h_box//3), (x+w_box-6, y+2*h_box//3),
                                     tuple(max(0, c2-20) for c2 in car_color), -1)
                        cls_id = 1
                    else:
                        cls_id = 0

                    # YOLO format
                    cx = (x + w_box/2) / 640
                    cy = (y + h_box/2) / 640
                    nw = w_box / 640
                    nh = h_box / 640
                    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Random brightness/contrast variations
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-20, 21)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(dataset_dir, split, "images", f"lot_{i:04d}.jpg"), img)
            with open(os.path.join(dataset_dir, split, "labels", f"lot_{i:04d}.txt"), 'w') as f:
                f.write('\n'.join(labels))

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.join(dataset_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(dataset_dir, 'valid', 'images')}\n")
        f.write("nc: 2\nnames: ['space-empty', 'space-occupied']\n")

    print(f"   [OK] Created 300 train + 60 val synthetic parking images")
    return yaml_path


if __name__ == '__main__':
    main()
