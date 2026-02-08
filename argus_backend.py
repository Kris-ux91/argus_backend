from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
import json
from datetime import datetime
import base64
from io import BytesIO
import hashlib

app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all for development

# =========================================
# YOUR CAMERA - UPDATE THIS LINE
RTSP_URL = "rtsp://admin:12345678@172.16.0.144:5543/live/channel0"
# =========================================

# Load YOLOv8 model (nano for speed, detects 80+ classes)
model = YOLO('yolov8n.pt')
model.fuse()  # Optimize speed

# Global state
latest_frame = None
latest_detections = []
threat_alerts = []
system_stats = {
    'fps': 0,
    'inference_time': 0,
    'total_detections': 0,
    'threats_today': 0,
    'gpu_usage': 0,
    'timestamp': time.time()
}
frame_count = 0
last_fps_time = time.time()

# Threat scoring configuration
THREAT_CLASSES = {
    'person': {'priority': 5, 'color': (0, 0, 255)},      # Red - High threat
    'knife': {'priority': 6, 'color': (0, 69, 255)},       # Orange - Critical
    'car': {'priority': 2, 'color': (0, 255, 0)},          # Green - Low
    'truck': {'priority': 3, 'color': (0, 255, 0)},
    'bicycle': {'priority': 1, 'color': (0, 255, 255)},    # Yellow
    'backpack': {'priority': 4, 'color': (0, 255, 255)},
    'handbag': {'priority': 4, 'color': (0, 255, 255)},
    'dog': {'priority': 1, 'color': (255, 0, 255)},        # Magenta - Animal
    'cat': {'priority': 1, 'color': (255, 0, 255)},
    'bird': {'priority': 0, 'color': (255, 255, 0)},       # Cyan - Ignore
}

def calculate_threat_score(detection, frame_shape):
    """Advanced threat scoring"""
    class_name = detection['class']
    conf = detection['confidence']
    x, y, w, h = detection['bbox']
    
    score = conf * 100  # Base score (0-100)
    
    # Object priority multiplier
    priority = THREAT_CLASSES.get(class_name, {'priority': 1})['priority']
    score *= (1 + priority * 0.2)
    
    # Proximity to bottom (fence area) - higher threat
    frame_height = frame_shape[0]
    proximity_factor = max(0, 1 - (y / frame_height))
    score += proximity_factor * 25
    
    # Size factor - larger objects more threatening
    size_factor = min(1.0, (w * h) / (frame_shape[0] * frame_shape[1] * 0.1))
    score += size_factor * 15
    
    # Loitering bonus (same object multiple frames) - track later
    score = min(100, score)
    return score

@app.route('/api/video_feed')
def video_feed():
    """Main MJPEG stream with detections"""
    def generate_frames():
        global frame_count, last_fps_time, latest_frame, latest_detections
        
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            # Fallback to webcam or demo
            cap = cv2.VideoCapture(0)
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                # Generate demo frame if no video
                frame = generate_demo_frame()
            
            frame_count += 1
            
            # YOLOv8 inference
            inference_start = time.time()
            results = model(frame, verbose=False, conf=0.5)
            inference_time = (time.time() - inference_start) * 1000
            
            # Process all detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        class_name = model.names[cls]
                        
                        # Calculate threat score
                        threat_score = calculate_threat_score({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': (x1, y1, x2-x1, y2-y1)
                        }, frame.shape)
                        
                        detection = {
                            'id': f"{frame_count}_{cls}_{conf:.2f}",
                            'class': class_name,
                            'confidence': conf,
                            'threat_score': threat_score,
                            'bbox': {'x': x1, 'y': y1, 'w': x2-x1, 'h': y2-y1},
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        detections.append(detection)
                        
                        # Draw on frame
                        if class_name in THREAT_CLASSES:
                            color = THREAT_CLASSES[class_name]['color']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(frame, f'{class_name} {conf:.1f}', 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(frame, f'Score:{threat_score:.0f}', 
                                      (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update global state
            latest_frame = frame.copy()
            latest_detections = detections[:10]  # Keep last 10
            
            # Auto-generate alerts for high threats
            for det in detections:
                if det['threat_score'] > 40:
                    alert = {
                        'id': hashlib.md5(f"{det['id']}{time.time()}".encode()).hexdigest()[:8],
                        'timestamp': datetime.now().isoformat(),
                        'threat_score': det['threat_score'],
                        'object_type': det['class'],
                        'confidence': det['confidence'],
                        'camera': 'Border Camera 1',
                        'bbox': det['bbox'],
                        'image_base64': encode_frame_to_base64(frame)
                    }
                    threat_alerts.append(alert)
                    if len(threat_alerts) > 100:  # Keep last 100
                        threat_alerts.pop(0)
            
            # Update stats
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                system_stats['fps'] = frame_count / (current_time - last_fps_time)
                system_stats['inference_time'] = inference_time
                system_stats['total_detections'] += len(detections)
                system_stats['threats_today'] = len([a for a in threat_alerts if a['threat_score'] > 40])
                frame_count = 0
                last_fps_time = current_time
            
            system_stats['timestamp'] = datetime.now().isoformat()
            
            # Stream frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_demo_frame():
    """Fallback demo frame when no camera"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add demo objects
    cv2.rectangle(frame, (150, 100), (300, 300), (0, 0, 255), 3)  # Person
    cv2.putText(frame, 'person 0.95', (160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.rectangle(frame, (400, 250), (550, 350), (0, 255, 0), 2)  # Car
    cv2.putText(frame, 'car 0.87', (410, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(frame, 'DEMO MODE - No Camera', (200, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

def encode_frame_to_base64(frame):
    """Encode frame for alert thumbnails"""
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/api/latest_detections')
def get_latest_detections():
    """Latest 10 detections for dashboard"""
    return jsonify(latest_detections)

@app.route('/api/threat_alerts')
def get_threat_alerts():
    """Recent threat alerts (last 50)"""
    recent = threat_alerts[-50:]
    return jsonify(recent)

@app.route('/api/system_stats')
def get_system_stats():
    """System performance metrics"""
    return jsonify(system_stats)

@app.route('/api/acknowledge_alert', methods=['POST'])
def acknowledge_alert():
    """Mark alert as acknowledged"""
    data = request.json
    alert_id = data.get('alert_id')
    
    # Mark as acknowledged (simplified)
    for alert in threat_alerts:
        if alert['id'] == alert_id:
            alert['acknowledged'] = True
            alert['acknowledged_time'] = datetime.now().isoformat()
            break
    
    return jsonify({'success': True})

@app.route('/api/screenshot')
def get_screenshot():
    """Latest frame for alert thumbnails"""
    global latest_frame
    if latest_frame is None:
        return jsonify({'error': 'No frame available'})
    
    img_base64 = encode_frame_to_base64(latest_frame)
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_base64}',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    """Health endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'camera': RTSP_URL,
        'model': 'YOLOv8n',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 ARGUS Surveillance Backend Starting...")
    print(f"📹 Camera: {RTSP_URL}")
    print("🌐 Backend: http://localhost:5000")
    print("📱 Video Feed: http://localhost:5000/api/video_feed")
    print("📊 Detections: http://localhost:5000/api/latest_detections")
    print("🔥 Keep this terminal running!\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
