"""Test the API endpoint directly."""
import requests
import json

with open('test_parking.jpg', 'rb') as f:
    files = {'image': ('test_parking.jpg', f, 'image/jpeg')}
    resp = requests.post('http://127.0.0.1:5000/api/detect', files=files)

data = resp.json()
print(f"Success: {data.get('success')}")
print(f"Total: {data.get('total_spaces')}")
print(f"Occupied: {data.get('occupied')}")
print(f"Available: {data.get('available')}")
print(f"Occupancy: {data.get('occupancy_rate')}%")
print(f"Confidence: {data.get('confidence_avg')}%")
print(f"Model: {data.get('model_info', {}).get('name')}")
print(f"Detections: {len(data.get('detections', []))}")
