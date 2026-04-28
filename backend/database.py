"""
ParkVision AI — SQLite Database Manager
Stores detection history for past analysis results.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'parkvision.db')


def get_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            original_image TEXT NOT NULL,
            annotated_image TEXT NOT NULL,
            heatmap_image TEXT,
            total_spaces INTEGER NOT NULL,
            occupied INTEGER NOT NULL,
            available INTEGER NOT NULL,
            occupancy_rate REAL NOT NULL,
            confidence_avg REAL NOT NULL,
            vehicle_types TEXT NOT NULL,
            zone_analysis TEXT,
            detections_json TEXT NOT NULL,
            processing_time REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_detection(data):
    """Save a detection result to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections 
        (timestamp, original_image, annotated_image, heatmap_image,
         total_spaces, occupied, available, occupancy_rate, confidence_avg,
         vehicle_types, zone_analysis, detections_json, processing_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['timestamp'],
        data['original_image'],
        data['annotated_image'],
        data.get('heatmap_image', ''),
        data['total_spaces'],
        data['occupied'],
        data['available'],
        data['occupancy_rate'],
        data['confidence_avg'],
        json.dumps(data['vehicle_types']),
        json.dumps(data.get('zone_analysis', {})),
        json.dumps(data['detections']),
        data['processing_time']
    ))
    conn.commit()
    detection_id = cursor.lastrowid
    conn.close()
    return detection_id


def get_all_detections():
    """Get all detection records, newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM detections ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append(_row_to_dict(row))
    return results


def get_detection(detection_id):
    """Get a single detection by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return _row_to_dict(row)
    return None


def delete_detection(detection_id):
    """Delete a detection record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT original_image, annotated_image, heatmap_image FROM detections WHERE id = ?',
                   (detection_id,))
    row = cursor.fetchone()

    if row:
        # Delete associated image files
        for img_field in ['original_image', 'annotated_image', 'heatmap_image']:
            img_path = row[img_field]
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except OSError:
                    pass

        cursor.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
        conn.commit()
        conn.close()
        return True

    conn.close()
    return False


def get_aggregate_stats():
    """Get aggregate statistics across all detections."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT 
            COUNT(*) as total_scans,
            COALESCE(SUM(occupied), 0) as total_vehicles_detected,
            COALESCE(AVG(occupancy_rate), 0) as avg_occupancy_rate,
            COALESCE(AVG(confidence_avg), 0) as avg_confidence,
            COALESCE(AVG(processing_time), 0) as avg_processing_time,
            COALESCE(SUM(total_spaces), 0) as total_spaces_analyzed
        FROM detections
    ''')
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'total_scans': row['total_scans'],
            'total_vehicles_detected': row['total_vehicles_detected'],
            'avg_occupancy_rate': round(row['avg_occupancy_rate'], 1),
            'avg_confidence': round(row['avg_confidence'], 1),
            'avg_processing_time': round(row['avg_processing_time'], 2),
            'total_spaces_analyzed': row['total_spaces_analyzed']
        }
    return {}


def clear_all():
    """Clear all detection records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM detections')
    conn.commit()
    conn.close()


def _row_to_dict(row):
    """Convert a database row to a dictionary."""
    return {
        'id': row['id'],
        'timestamp': row['timestamp'],
        'original_image': row['original_image'],
        'annotated_image': row['annotated_image'],
        'heatmap_image': row['heatmap_image'],
        'total_spaces': row['total_spaces'],
        'occupied': row['occupied'],
        'available': row['available'],
        'occupancy_rate': row['occupancy_rate'],
        'confidence_avg': row['confidence_avg'],
        'vehicle_types': json.loads(row['vehicle_types']),
        'zone_analysis': json.loads(row['zone_analysis']) if row['zone_analysis'] else {},
        'detections': json.loads(row['detections_json']),
        'processing_time': row['processing_time']
    }


# Initialize database on import
init_db()
