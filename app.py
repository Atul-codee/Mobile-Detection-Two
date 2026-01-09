from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import pygame
import time
import os

# --- FLASK APP INITIALIZE ---
app = Flask(__name__)

# --- CONFIGURATION  ---
AUDIO_FILE = "audio.mp3"  
CONFIDENCE_LEVEL = 0.5    
COOLDOWN_SECONDS = 11     
FRAME_SKIP = 3  
# ---------------------

if not os.path.exists(AUDIO_FILE):
    print(f"ERROR: '{AUDIO_FILE}' file nahi mili!")
    # Flask e exit() use kora thik na, tai just print rakhlam

# Audio setup
pygame.mixer.init()
try:
    pygame.mixer.music.load(AUDIO_FILE)
except Exception as e:
    print(f"Audio Error: {e}")

print("Loading Fast AI Model...")
model = YOLO('yolov8n.pt')

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# Global variables for state
last_alert_time = 0
frame_count = 0

def generate_frames():
    global last_alert_time, frame_count
    
    # State save karne ke liye (Loop ke bahar taki flicker na ho)
    phone_detected = False 
    boxes_to_draw = [] 

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1

        # --- SPEED TRICK: FRAME SKIPPING ---
        if frame_count % FRAME_SKIP == 0:
            results = model(frame, stream=True, verbose=False, conf=CONFIDENCE_LEVEL, imgsz=640)
            
            phone_detected = False
            boxes_to_draw = [] 

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    if class_name == 'cell phone':
                        phone_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes_to_draw.append((x1, y1, x2, y2))
        
        # --- DRAWING ---
        # Phone detect hole draw koro
        if len(boxes_to_draw) > 0: # Check if list is not empty
            # Current logic e jodi frame skip hoy, tao ager detected box draw hobe (simple logic)
             for (x1, y1, x2, y2) in boxes_to_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE RAKHHHH NEECHE!", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Audio Logic (Server side e sound hobe)
             current_time = time.time()
             if current_time - last_alert_time > COOLDOWN_SECONDS:
                print(">>> ALERT: Fast Detection!")
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                last_alert_time = current_time

        # --- WEB DISPLAY ENCODING ---
        # Cv2.imshow er bodole image ke jpg te convert kore web e pathate hobe
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Web browser ke frame pathano (Yield)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- FLASK ROUTES (API) ---

@app.route('/')
def index():
    """Website er main page show korbe"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video stream handle korbe"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Debug mode True thakle code change korle auto restart hobe
    app.run(debug=True, port=5000)