import os
import sys
import logging
import cv2
import cvzone
from flask import Flask, Response
from flask_cors import CORS
from ultralytics import YOLO

# Disable Flask console noise
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "final_model.pt")

# Define your two video paths
video_path_1 = os.path.join(base_dir, "Parking_Lot_CCTV_1.mp4") 
video_path_2 = os.path.join(base_dir, "Parking_Lot_CCTV_2.1.mp4") 

classNames = ['occupied_slot', 'free_slot']

# --- Load Model Once (Shared across all feeds) ---
print("\n--- Loading Shared YOLO Model ---")
try:
    model = YOLO(model_path)
    model.to('cpu')
    print("✅ Model Loaded into Memory.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

def stream_logic(path):
    """Generic logic to process a video path and yield frames."""
    cap = cv2.VideoCapture(path)
    
    while True:
        success, img = cap.read()
        
        # Loop the video if it ends
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Use the shared model for inference
        results = model(img, stream=True, verbose=False, conf=0.4)
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = classNames[cls] if cls < len(classNames) else "Unknown"
                color = (0, 255, 0) if label == 'free_slot' else (0, 0, 255)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                # Draw corner rectangles
                cvzone.cornerRect(img, (x1, y1, w, h), t=2, rt=0, colorC=color, colorR=color)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
            
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Routes ---

@app.route('/video_feed_1')
def video_feed_1():
    """Route for the first video stream."""
    return Response(stream_logic(video_path_1), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    """Route for the second video stream."""
    return Response(stream_logic(video_path_2), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("\n🚀 Multi-Stream AI Server Active")
    print(f"🔗 Feed 1: http://localhost:5000/video_feed_1")
    print(f"🔗 Feed 2: http://localhost:5000/video_feed_2")
    
    # threaded=True allows Flask to handle both generators at the same time

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
