# -*- coding: utf-8 -*-
"""Test the Smart Detection engine."""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
os.makedirs('test_output', exist_ok=True)

from parking_detector import ParkingDetector
det = ParkingDetector()

result = det.analyze('test_parking.jpg', 'test_output')

print(f"\n{'='*50}")
print(f"  DETECTION RESULTS")
print(f"{'='*50}")
print(f"  Total Spaces: {result['total_spaces']}")
print(f"  Occupied:     {result['occupied']}")
print(f"  Available:    {result['available']}")
print(f"  Occupancy:    {result['occupancy_rate']}%")
print(f"  Avg Conf:     {result['confidence_avg']}%")
print(f"  Model:        {result['model_info']['name']}")
print(f"  Vehicles:     {result['vehicle_types']}")
print(f"\n  Top 10 detections:")
for d in sorted(result['detections'], key=lambda x: x['confidence'], reverse=True)[:10]:
    print(f"    - {d['class_name']} ({d['confidence']:.0%})")
print(f"{'='*50}")
