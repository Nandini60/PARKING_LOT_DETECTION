"""
ParkVision AI — Flask Backend Server
REST API for parking lot detection and analysis.
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from parking_detector import ParkingDetector
from database import save_detection, get_all_detections, get_detection, delete_detection, get_aggregate_stats
import os
import uuid
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
SAMPLE_FOLDER = os.path.join(os.path.dirname(__file__), 'sample_images')
FRONTEND_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)

# Initialize detector (loads YOLOv8 model)
print("[ParkVision] Initializing AI engine...")
detector = ParkingDetector(confidence_threshold=0.40)
print("[ParkVision] AI engine ready!")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== FRONTEND SERVING ====================

@app.route('/')
def serve_frontend():
    """Serve the main frontend page."""
    return send_from_directory(FRONTEND_FOLDER, 'index.html')


@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files."""
    return send_from_directory(os.path.join(FRONTEND_FOLDER, 'css'), filename)


@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files."""
    return send_from_directory(os.path.join(FRONTEND_FOLDER, 'js'), filename)


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets."""
    return send_from_directory(os.path.join(FRONTEND_FOLDER, 'assets'), filename)


# ==================== API ROUTES ====================

@app.route('/api/detect', methods=['POST'])
def detect_parking():
    """
    Upload an image and run parking lot detection.
    Returns JSON results with annotated image paths.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        # Save uploaded file
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Check file size
        if os.path.getsize(filepath) > MAX_FILE_SIZE:
            os.remove(filepath)
            return jsonify({'error': 'File too large. Maximum size: 20MB'}), 400

        # Run AI detection
        result = detector.analyze(filepath, RESULTS_FOLDER)

        # Save to database
        db_data = {
            'timestamp': datetime.now().isoformat(),
            'original_image': filepath,
            'annotated_image': os.path.join(RESULTS_FOLDER, result['annotated_image']),
            'heatmap_image': os.path.join(RESULTS_FOLDER, result['heatmap_image']),
            'total_spaces': result['total_spaces'],
            'occupied': result['occupied'],
            'available': result['available'],
            'occupancy_rate': result['occupancy_rate'],
            'confidence_avg': result['confidence_avg'],
            'vehicle_types': result['vehicle_types'],
            'zone_analysis': result['zone_analysis'],
            'detections': result['detections'],
            'processing_time': result['processing_time']
        }
        detection_id = save_detection(db_data)

        # Build response
        response = {
            'success': True,
            'id': detection_id,
            'timestamp': db_data['timestamp'],
            'total_spaces': result['total_spaces'],
            'occupied': result['occupied'],
            'available': result['available'],
            'occupancy_rate': result['occupancy_rate'],
            'confidence_avg': result['confidence_avg'],
            'vehicle_types': result['vehicle_types'],
            'zone_analysis': result['zone_analysis'],
            'detections': result['detections'],
            'annotated_image': f'/api/images/results/{result["annotated_image"]}',
            'heatmap_image': f'/api/images/results/{result["heatmap_image"]}',
            'original_image': f'/api/images/uploads/{filename}',
            'processing_time': result['processing_time'],
            'image_dimensions': result['image_dimensions'],
            'model_info': result['model_info']
        }

        return jsonify(response), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@app.route('/api/images/results/<filename>')
def serve_result_image(filename):
    """Serve annotated result images."""
    return send_from_directory(RESULTS_FOLDER, filename)


@app.route('/api/images/uploads/<filename>')
def serve_upload_image(filename):
    """Serve uploaded images."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/api/images/samples/<filename>')
def serve_sample_image(filename):
    """Serve sample images."""
    return send_from_directory(SAMPLE_FOLDER, filename)


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all detection history."""
    try:
        detections = get_all_detections()
        # Convert file paths to API URLs
        for det in detections:
            det['original_image'] = f'/api/images/uploads/{os.path.basename(det["original_image"])}'
            det['annotated_image'] = f'/api/images/results/{os.path.basename(det["annotated_image"])}'
            if det['heatmap_image']:
                det['heatmap_image'] = f'/api/images/results/{os.path.basename(det["heatmap_image"])}'
        return jsonify({'success': True, 'detections': detections}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<int:detection_id>', methods=['GET'])
def get_history_item(detection_id):
    """Get a specific detection by ID."""
    try:
        det = get_detection(detection_id)
        if det:
            det['original_image'] = f'/api/images/uploads/{os.path.basename(det["original_image"])}'
            det['annotated_image'] = f'/api/images/results/{os.path.basename(det["annotated_image"])}'
            if det['heatmap_image']:
                det['heatmap_image'] = f'/api/images/results/{os.path.basename(det["heatmap_image"])}'
            return jsonify({'success': True, 'detection': det}), 200
        return jsonify({'error': 'Detection not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<int:detection_id>', methods=['DELETE'])
def delete_history_item(detection_id):
    """Delete a detection record."""
    try:
        success = delete_detection(detection_id)
        if success:
            return jsonify({'success': True, 'message': 'Detection deleted'}), 200
        return jsonify({'error': 'Detection not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get aggregate statistics."""
    try:
        stats = get_aggregate_stats()
        return jsonify({'success': True, 'stats': stats}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'ParkVision AI',
        'model': 'YOLOv8 Nano',
        'version': '1.0.0'
    }), 200


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  🅿️  ParkVision AI — Smart Parking Detector")
    print("  🌐  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
