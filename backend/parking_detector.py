"""
ParkVision AI — Parking Lot Detection Engine v6 (Best of Both Models)
=====================================================================
PRIMARY: Custom trained model (50 epochs, 1500 images) at conf=0.40
         → Detects parking spaces (empty/occupied) from aerial views
BOOST:   Generic YOLOv8m for vehicle validation on detected occupied spaces

The custom model IS accurate — the key is using conf >= 0.40 to cut noise.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import os, time, math

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

CUSTOM_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'parking_best.pt')
)


class ParkingDetector:
    def __init__(self, confidence_threshold=0.40):
        self.custom_mode = False

        # Load custom trained model (primary — trained on parking data)
        if os.path.exists(CUSTOM_MODEL_PATH):
            print(f"[ParkVision] Loading custom model: {CUSTOM_MODEL_PATH}")
            self.model = YOLO(CUSTOM_MODEL_PATH)
            self.custom_mode = True
            print(f"[ParkVision] Custom model classes: {self.model.names}")
        else:
            print(f"[ParkVision] No custom model found, using generic")
            self.model = YOLO('yolov8m.pt')

        # Load generic model (for vehicle validation boost)
        print("[ParkVision] Loading generic YOLOv8m for vehicle validation...")
        self.generic_model = YOLO('yolov8m.pt')

        self.conf = confidence_threshold
        print(f"[ParkVision] Engine ready! conf={self.conf}, custom={self.custom_mode}")

    def analyze(self, image_path, output_dir):
        t0 = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        original = image.copy()
        h, w = image.shape[:2]
        result = self._detect(image, original, w, h, output_dir)
        result['processing_time'] = round(time.time() - t0, 3)
        result['image_dimensions'] = {'width': w, 'height': h}
        return result

    def _detect(self, image, original, w, h, output_dir):
        """
        Detection pipeline:
        1. Run custom model at conf=0.40 (high accuracy, no noise)
        2. Run on enhanced + high-res versions for more coverage
        3. NMS to merge duplicates
        4. Validate with generic model to boost confidence
        """

        # ── STEP 1: Custom model multi-pass ──
        all_boxes, all_scores, all_types = [], [], []

        # Pass 1: Original @ 640
        self._run_custom(image, 640, all_boxes, all_scores, all_types)

        # Pass 2: Enhanced
        enhanced = self._enhance(image)
        self._run_custom(enhanced, 640, all_boxes, all_scores, all_types)

        # Pass 3: High-res
        if max(h, w) >= 600:
            self._run_custom(image, 1280, all_boxes, all_scores, all_types)

        # Pass 4: CLAHE
        clahe_img = self._apply_clahe(image)
        self._run_custom(clahe_img, 640, all_boxes, all_scores, all_types)

        # NMS merge
        detections = self._nms(all_boxes, all_scores, all_types) if all_boxes else []

        # ── STEP 2: Generic model vehicle validation ──
        vehicles = self._detect_vehicles_generic(image, w, h)

        # Boost occupied detections that overlap with generic vehicles
        for det in detections:
            if det['space_type'] == 'occupied':
                for v in vehicles:
                    if self._iou(det['bbox'], v['bbox']) > 0.1:
                        det['confidence'] = min(0.99, det['confidence'] * 1.15)
                        det['validated'] = True
                        break

        # Any generic vehicle not matching a custom detection -> add as occupied
        for v in vehicles:
            matched = any(self._iou(v['bbox'], d['bbox']) > 0.1 for d in detections)
            if not matched and v['confidence'] > 0.3:
                detections.append({
                    'bbox': v['bbox'], 'space_type': 'occupied',
                    'confidence': v['confidence'],
                    'center': v['center'], 'area': v['area'],
                    'validated': True
                })

        # ── STEP 3: TEXTURE VALIDATION ──
        # Real cars have complex textures (glass, metal, shadows)
        # Empty pavement is uniform — reclassify false "occupied" as "empty"
        for det in detections:
            if det['space_type'] == 'occupied' and not det.get('validated'):
                x1, y1, x2, y2 = det['bbox']
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                rw, rh = x2c - x1c, y2c - y1c
                if rw < 10 or rh < 10:
                    continue

                roi = image[y1c:y2c, x1c:x2c]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Crop CENTER 60% of ROI (ignore parking lines at edges)
                margin_x, margin_y = int(rw * 0.2), int(rh * 0.2)
                center_roi = roi[margin_y:rh-margin_y, margin_x:rw-margin_x]
                center_gray = gray_roi[margin_y:rh-margin_y, margin_x:rw-margin_x]
                if center_gray.size < 100:
                    center_gray = gray_roi
                    center_roi = roi

                # 1. Edge density (center) — cars have many internal edges
                lap = cv2.Laplacian(center_gray, cv2.CV_64F)
                edge_score = np.mean(np.abs(lap))

                # 2. Color variance (center) — cars have varied colors
                color_std = np.mean([np.std(center_roi[:,:,c]) for c in range(3)])

                # 3. Gradient magnitude (center)
                gx = cv2.Sobel(center_gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(center_gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.mean(np.sqrt(gx**2 + gy**2))

                # 4. Brightness uniformity — empty pavement is very uniform
                brightness_std = np.std(center_gray.astype(np.float32))

                # 5. Histogram entropy — cars have complex color distributions
                hist = cv2.calcHist([center_gray], [0], None, [32], [0, 256])
                hist = hist.flatten() / max(hist.sum(), 1)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))

                # Combined score: higher = more likely a real car
                texture_score = (
                    edge_score * 0.25 +
                    color_std * 0.20 +
                    grad_mag * 0.20 +
                    brightness_std * 0.20 +
                    entropy * 3.0 * 0.15  # Scale entropy to similar range
                )

                # Decision: confidence-aware thresholds
                is_empty = False
                conf = det['confidence']

                if conf >= 0.92:
                    # Very high conf: only flip if EXTREMELY uniform
                    if brightness_std < 10 and color_std < 12 and edge_score < 5:
                        is_empty = True
                elif conf >= 0.75:
                    # High conf: flip if texture is low
                    if texture_score < 20:
                        is_empty = True
                    elif brightness_std < 13 and color_std < 16 and edge_score < 7:
                        is_empty = True
                elif conf >= 0.55:
                    # Medium: more aggressive
                    if texture_score < 28:
                        is_empty = True
                    elif brightness_std < 16 and color_std < 20:
                        is_empty = True
                else:
                    # Low: very aggressive
                    if texture_score < 32:
                        is_empty = True
                    elif brightness_std < 20 or color_std < 22:
                        is_empty = True
                    elif edge_score < 10 and grad_mag < 35:
                        is_empty = True

                if is_empty:
                    det['space_type'] = 'empty'
                    det['confidence'] = max(0.50, min(0.95, 1.0 - (texture_score / 60)))

        # ── STEP 3b: REFERENCE-BASED COMPARISON ──
        # Use confirmed empty spaces as baseline, compare remaining occupied against them
        empty_features = []
        for det in detections:
            if det['space_type'] == 'empty':
                x1, y1, x2, y2 = det['bbox']
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                if x2c - x1c < 10 or y2c - y1c < 10:
                    continue
                roi = image[y1c:y2c, x1c:x2c]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                empty_features.append({
                    'mean_brightness': np.mean(gray),
                    'brightness_std': np.std(gray.astype(np.float32)),
                    'mean_color': np.mean(roi, axis=(0, 1)),
                    'color_std': np.mean([np.std(roi[:,:,c]) for c in range(3)])
                })

        if len(empty_features) >= 3:
            # Build baseline from confirmed empties
            avg_brightness = np.mean([f['mean_brightness'] for f in empty_features])
            avg_bright_std = np.mean([f['brightness_std'] for f in empty_features])
            avg_color = np.mean([f['mean_color'] for f in empty_features], axis=0)
            avg_color_std = np.mean([f['color_std'] for f in empty_features])

            for det in detections:
                if det['space_type'] == 'occupied' and not det.get('validated'):
                    x1, y1, x2, y2 = det['bbox']
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w, x2), min(h, y2)
                    if x2c - x1c < 10 or y2c - y1c < 10:
                        continue

                    roi = image[y1c:y2c, x1c:x2c]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # Compare this space to the empty baseline
                    mb = np.mean(gray)
                    ms = np.std(gray.astype(np.float32))
                    mc = np.mean(roi, axis=(0, 1))
                    cs = np.mean([np.std(roi[:,:,c]) for c in range(3)])

                    # Color distance from empty baseline
                    color_dist = np.linalg.norm(mc - avg_color)

                    # Brightness similarity
                    bright_diff = abs(mb - avg_brightness)

                    # Looks like empty baseline?
                    looks_like_empty = (
                        color_dist < 30 and         # Similar color to empties
                        bright_diff < 25 and         # Similar brightness
                        ms < avg_bright_std * 1.8 and  # Not much more textured
                        cs < avg_color_std * 1.8     # Not much more colorful
                    )

                    if looks_like_empty and det['confidence'] < 0.88:
                        det['space_type'] = 'empty'
                        det['confidence'] = max(0.55, 0.90 - color_dist / 50)

        # ── STEP 3c: SPATIAL CONSISTENCY ──
        # If a low-conf occupied space is surrounded by empty spaces, flip it
        for i, det in enumerate(detections):
            if det['space_type'] == 'occupied' and det['confidence'] < 0.80 and not det.get('validated'):
                cx, cy = det['center']
                avg_size = max(50, int(np.mean([max(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]) for d in detections])))
                radius = avg_size * 2.5
                nearby_empty = 0
                nearby_occupied = 0
                for j, other in enumerate(detections):
                    if i == j:
                        continue
                    ox, oy = other['center']
                    dist = math.sqrt((cx - ox)**2 + (cy - oy)**2)
                    if dist < radius:
                        if other['space_type'] == 'empty':
                            nearby_empty += 1
                        else:
                            nearby_occupied += 1
                if nearby_empty >= 2 and nearby_empty > nearby_occupied:
                    det['space_type'] = 'empty'
                    det['confidence'] = max(0.55, det['confidence'])

        # ── STEP 4: Count ──
        empty = sum(1 for d in detections if d['space_type'] == 'empty')
        occupied = sum(1 for d in detections if d['space_type'] == 'occupied')
        total = empty + occupied

        vtypes = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        for v in vehicles:
            if v.get('class_name') in vtypes:
                vtypes[v['class_name']] += 1
        # Ensure car count >= occupied if no specific vehicle types
        if sum(vtypes.values()) == 0:
            vtypes['car'] = occupied

        occ_rate = round((occupied / max(total, 1)) * 100, 1)
        avg_conf = round(
            sum(d['confidence'] for d in detections) / max(len(detections), 1) * 100, 1
        )

        # ── STEP 4: Build output ──
        det_list = []
        for d in detections:
            det_list.append({
                'bbox': d['bbox'], 'class_id': 0,
                'class_name': 'occupied' if d['space_type'] == 'occupied' else 'empty',
                'confidence': d['confidence'],
                'center': d['center'], 'area': d['area']
            })

        # Annotate
        ann = self._annotate(original.copy(), detections, total, empty, occ_rate)
        ann_fn = f"annotated_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(output_dir, ann_fn), ann, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Heatmap
        occ_dets = [d for d in detections if d['space_type'] == 'occupied']
        hm = self._heatmap(original.copy(), occ_dets, w, h)
        hm_fn = f"heatmap_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(output_dir, hm_fn), hm, [cv2.IMWRITE_JPEG_QUALITY, 95])

        zones = self._zones(detections, w, h, total)

        return {
            'total_spaces': total, 'occupied': occupied,
            'available': empty, 'occupancy_rate': occ_rate,
            'confidence_avg': avg_conf, 'vehicle_types': vtypes,
            'zone_analysis': zones, 'detections': det_list,
            'annotated_image': ann_fn, 'heatmap_image': hm_fn,
            'model_info': {
                'name': 'ParkVision Custom+Generic (50-epoch trained)',
                'confidence_threshold': self.conf,
                'total_detections_raw': len(detections),
                'vehicles_generic': len(vehicles)
            }
        }

    # ────────────────────────────────────────────────
    #  CUSTOM MODEL INFERENCE
    # ────────────────────────────────────────────────
    def _run_custom(self, img, imgsz, boxes, scores, types):
        """Run custom model and collect detections."""
        results = self.model(img, conf=self.conf, iou=0.45, imgsz=imgsz, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid = int(box.cls[0])
                name = self.model.names.get(cid, '').lower()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                space_type = 'occupied'
                if 'empty' in name or 'free' in name:
                    space_type = 'empty'

                conf = round(float(box.conf[0]), 4)
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(conf)
                types.append(space_type)

    # ────────────────────────────────────────────────
    #  GENERIC MODEL VEHICLE DETECTION
    # ────────────────────────────────────────────────
    def _detect_vehicles_generic(self, image, w, h):
        """Detect real vehicles using generic YOLOv8m — used to validate."""
        vehicles = []
        for img_in, sz in [(image, 640), (self._enhance(image), 640)]:
            results = self.generic_model(img_in, conf=0.15, iou=0.5, imgsz=sz, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cid = int(box.cls[0])
                    if cid in VEHICLE_CLASSES:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        vehicles.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class_name': VEHICLE_CLASSES[cid],
                            'confidence': round(float(box.conf[0]), 4),
                            'center': [int((x1+x2)/2), int((y1+y2)/2)],
                            'area': int((x2-x1)*(y2-y1))
                        })

        # NMS on vehicles
        if len(vehicles) > 1:
            vb = [v['bbox'] for v in vehicles]
            vs = [v['confidence'] for v in vehicles]
            bxywh = [(b[0], b[1], b[2]-b[0], b[3]-b[1]) for b in vb]
            idx = cv2.dnn.NMSBoxes(bxywh, vs, 0.15, 0.4)
            if len(idx) > 0:
                if isinstance(idx, np.ndarray):
                    idx = idx.flatten()
                vehicles = [vehicles[i] for i in idx]

        return vehicles

    # ────────────────────────────────────────────────
    #  NMS
    # ────────────────────────────────────────────────
    def _nms(self, boxes, scores, types, iou_thr=0.3):
        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        bxywh = [(int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])) for b in boxes_np]
        indices = cv2.dnn.NMSBoxes(bxywh, scores_np.tolist(), self.conf, iou_thr)
        result = []
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            for i in indices:
                x1, y1, x2, y2 = boxes[i]
                result.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'space_type': types[i],
                    'confidence': round(float(scores_np[i]), 4),
                    'center': [int((x1+x2)/2), int((y1+y2)/2)],
                    'area': int((x2-x1)*(y2-y1))
                })
        return result

    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / max(a1 + a2 - inter, 1e-6)

    # ────────────────────────────────────────────────
    #  IMAGE ENHANCEMENT
    # ────────────────────────────────────────────────
    def _enhance(self, image):
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Contrast(pil).enhance(1.5)
        pil = ImageEnhance.Sharpness(pil).enhance(1.6)
        pil = ImageEnhance.Brightness(pil).enhance(1.15)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ────────────────────────────────────────────────
    #  ANNOTATION
    # ────────────────────────────────────────────────
    def _annotate(self, image, dets, total, avail, occ_rate):
        h, w = image.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            is_empty = d['space_type'] == 'empty'
            color = (0, 220, 0) if is_empty else (0, 80, 255)

            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{'EMPTY' if is_empty else 'OCCUPIED'} {d['confidence']:.0%}"
            fs = 0.45
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(image, (x1, y1-lh-8), (x1+lw+6, y1), color, -1)
            cv2.putText(image, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1)

        # Info bar
        ph = 80
        panel = np.zeros((ph, w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "PARKVISION AI - TRAINED MODEL (50 EPOCHS)", (15, 25), f, 0.6, (100, 200, 255), 2)
        sy = 55
        occ = total - avail
        cv2.putText(panel, f"TOTAL: {total}", (15, sy), f, 0.55, (255,255,255), 1)
        cv2.putText(panel, f"OCCUPIED: {occ}", (200, sy), f, 0.55, (0,80,255), 1)
        cv2.putText(panel, f"AVAILABLE: {avail}", (420, sy), f, 0.55, (0,220,0), 1)
        bx, bw = 640, min(200, w - 660)
        if bx + bw < w:
            cv2.putText(panel, f"OCCUPANCY: {occ_rate}%", (bx, sy-20), f, 0.45, (200,200,200), 1)
            cv2.rectangle(panel, (bx, sy-10), (bx+bw, sy+5), (60,60,60), -1)
            fw = int(bw * (occ_rate / 100))
            bc = (0,220,0) if occ_rate < 70 else (0,165,255) if occ_rate < 90 else (0,0,255)
            cv2.rectangle(panel, (bx, sy-10), (bx+fw, sy+5), bc, -1)
        return np.vstack([panel, image])

    # ────────────────────────────────────────────────
    #  ZONES
    # ────────────────────────────────────────────────
    def _zones(self, dets, w, h, total):
        mx, my = w//2, h//2
        zones = {
            'top_left': {'name': 'Zone A', 'occupied': 0, 'empty': 0},
            'top_right': {'name': 'Zone B', 'occupied': 0, 'empty': 0},
            'bottom_left': {'name': 'Zone C', 'occupied': 0, 'empty': 0},
            'bottom_right': {'name': 'Zone D', 'occupied': 0, 'empty': 0},
        }
        for d in dets:
            cx, cy = d['center']
            key = ('top_' if cy < my else 'bottom_') + ('left' if cx < mx else 'right')
            if d['space_type'] == 'occupied':
                zones[key]['occupied'] += 1
            else:
                zones[key]['empty'] += 1
        for z in zones.values():
            z['total'] = z['occupied'] + z['empty']
            z['available'] = z['empty']
            z['occupancy'] = round((z['occupied'] / max(z['total'], 1)) * 100, 1)
        return zones

    # ────────────────────────────────────────────────
    #  HEATMAP
    # ────────────────────────────────────────────────
    def _heatmap(self, image, dets, w, h):
        hm = np.zeros((h, w), dtype=np.float32)
        for d in dets:
            cx, cy = d['center']
            r = max(int(math.sqrt(d['area']) * 0.8), 30)
            yc, xc = np.ogrid[-r:r+1, -r:r+1]
            g = np.exp(-(xc**2 + yc**2) / (2*(r/2)**2))
            ys, ye = max(0,cy-r), min(h,cy+r+1)
            xs, xe = max(0,cx-r), min(w,cx+r+1)
            gys, gxs = max(0,r-cy), max(0,r-cx)
            gye, gxe = gys+(ye-ys), gxs+(xe-xs)
            if ye > ys and xe > xs:
                hm[ys:ye, xs:xe] += g[gys:gye, gxs:gxe]
        if hm.max() > 0:
            hm = (hm / hm.max() * 255).astype(np.uint8)
        else:
            hm = hm.astype(np.uint8)
        colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        result = cv2.addWeighted(image, 0.5, colored, 0.5, 0)
        cv2.putText(result, "OCCUPANCY DENSITY HEATMAP", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        return result
