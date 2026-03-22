import cv2
import os
import time
import threading
import numpy as np
from ultralytics import YOLO

# Optional Twilio Alert System
try:
    import keys
    from twilio.rest import Client
    TWILIO_ENABLED = True
except ImportError:
    print("Please Create Keys.py file with your Twilio Credentials")
    exit()
   

# Model Paths
OBJ_MODEL_PATH = "yolov8n.pt"# place Path of your Object detection model. Use .pt file for the path 
FIRE_MODEL_PATH = "fire_model.pt" # place Path of your Fire/Smoke detection model. Use .pt file for the path 

if not os.path.exists(FIRE_MODEL_PATH):
    raise FileNotFoundError(f"Missing: {FIRE_MODEL_PATH}")

try:
    model_obj = YOLO(OBJ_MODEL_PATH)
    model_fire = YOLO(FIRE_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

# Detection Thresholds
SMALL_THRESH = 3000
LARGE_THRESH = 15000
REQUIRED_STREAK = 3

current_streak = 0
sms_cooldown = 60
last_sms_time = 0
last_alert_level = 0

# Performance Settings
FPS_TARGET = 15.0
INFERENCE_INTERVAL = 1.0 / FPS_TARGET

# Alert System
def trigger_alert(message, level):
    global last_sms_time, last_alert_level

    now = time.time()
    is_escalation = (level == 3 and last_alert_level < 3)

    if (now - last_sms_time > sms_cooldown) or is_escalation:
        last_sms_time = now
        last_alert_level = level

        print(f"\n🚨 ALERT: {message}")

        if TWILIO_ENABLED:
            threading.Thread(target=_send_sms, args=(message,), daemon=True).start()
        else:
            print("⚠️ Twilio disabled. SMS simulated.\n")


def _send_sms(msg):
    try:
        client = Client(keys.accountSID, keys.authToken)

        client.messages.create(
            body=f"YOLO Early FIRE/SMOKE detection and Alert System: {msg}",
            from_=keys.twilioNumber,
            to=keys.targetNumber
        )

        print("✅ SMS SENT\n")

    except Exception as e:
        print(f"❌ SMS FAILED: {e}\n")

# HSV Fire Validation
def is_physically_fire(frame, box):

    x1, y1, x2, y2 = map(int, box)

    roi = frame[max(0, y1):y2, max(0, x1):x2]

    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_fire = np.array([0, 100, 200])
    upper_fire = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    ratio = cv2.countNonZero(mask) / float(roi.shape[0] * roi.shape[1] + 1)

    return ratio > 0.02

# Bounding Box Intersection
def intersects(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return not (
        ax2 < bx1 or
        ax1 > bx2 or
        ay2 < by1 or
        ay1 > by2
    )

# Main System
def main():

    global current_streak, last_alert_level

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_frame_time = 0
    last_inf = 0

    c_boxes = []
    c_p_boxes = []
    c_f_boxes = []
    c_s_boxes = []
    c_i_boxes = []

    alert_text = "SYSTEM SAFE"
    alert_color = (0, 255, 0)

    detected_labels = []

    print("🔥 Fire Detection System Started")
    print("Press Q to quit\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0

        prev_frame_time = new_frame_time

        # Controlled Inference
        if new_frame_time - last_inf >= INFERENCE_INTERVAL:

            last_inf = new_frame_time

            c_boxes.clear()
            c_p_boxes.clear()
            c_f_boxes.clear()
            c_s_boxes.clear()
            c_i_boxes.clear()
            detected_labels.clear()

            max_fire_area = 0
            primary_fire_box = None
            person_near = False

            # Object Detection
            obj_results = model_obj(frame, verbose=False)[0]

            for box in obj_results.boxes:

                cls = int(box.cls[0])
                label = model_obj.names[cls]

                detected_labels.append(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 0:
                    c_p_boxes.append((x1, y1, x2, y2))
                else:
                    c_boxes.append((x1, y1, x2, y2))

            # Fire Detection
            fire_results = model_fire(frame, verbose=False, conf=0.65)[0]

            valid_hazard_found_this_frame = False

            for box in fire_results.boxes:

                label = model_fire.names[int(box.cls[0])].lower()

                coords = box.xyxy[0]

                x1, y1, x2, y2 = map(int, coords)

                if label in ["fire", "smoke", "flame"]:

                    is_valid = False

                    if label == "smoke":

                        is_valid = True
                        c_s_boxes.append((x1, y1, x2, y2))

                    else:

                        if is_physically_fire(frame, coords):

                            is_valid = True
                            c_f_boxes.append((x1, y1, x2, y2))

                        else:

                            c_i_boxes.append((x1, y1, x2, y2))

                    if is_valid:

                        valid_hazard_found_this_frame = True

                        area = (x2 - x1) * (y2 - y1)

                        if area > max_fire_area:
                            max_fire_area = area
                            primary_fire_box = (x1, y1, x2, y2)

            # Temporal Hazard Verification
            if valid_hazard_found_this_frame:

                current_streak += 1

                if current_streak > REQUIRED_STREAK + 5:
                    current_streak = REQUIRED_STREAK + 5

            else:

                current_streak = max(0, current_streak - 1)

                if current_streak == 0:
                    last_alert_level = 0

            # Human Proximity Check
            if primary_fire_box:

                for p_box in c_p_boxes:

                    if intersects(p_box, primary_fire_box):

                        person_near = True

            # Alert Logic
            alert_text = "SYSTEM SAFE"
            alert_color = (0, 255, 0)

            if current_streak >= REQUIRED_STREAK:

                if max_fire_area >= LARGE_THRESH:

                    alert_text = "LEVEL 3: CRITICAL EMERGENCY"
                    alert_color = (0, 0, 255)

                    trigger_alert(
                        "LEVEL 3: Large hazard detected. Evacuate immediately.",
                        3
                    )

                elif person_near:

                    alert_text = "LEVEL 1: Hazard Near Person"
                    alert_color = (0, 255, 255)

                    trigger_alert(
                        "Level 1: Hazard detected near human presence.",
                        1
                    )

                else:

                    alert_text = "LEVEL 2: Unattended Hazard"
                    alert_color = (0, 165, 255)

                    trigger_alert(
                        "Level 2: Unattended hazard detected.",
                        2
                    )

            elif current_streak > 0:

                alert_text = f"VERIFYING HAZARD ({current_streak}/{REQUIRED_STREAK})"
                alert_color = (0, 100, 255)

        # Draw Bounding Boxes
        for x1, y1, x2, y2 in c_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        for x1, y1, x2, y2 in c_p_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for x1, y1, x2, y2 in c_f_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "CONFIRMED FIRE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for x1, y1, x2, y2 in c_s_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 3)
            cv2.putText(frame, "SMOKE DETECTED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        for x1, y1, x2, y2 in c_i_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
            cv2.putText(frame, "IGNORED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)

        # Diagnostics Panel
        h, w = frame.shape[:2]

        panel = np.zeros((h, 220, 3), dtype=np.uint8)

        cv2.putText(panel, "DIAGNOSTICS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        fps_color = (0, 255, 0) if fps > 10 else (0, 165, 255)

        cv2.putText(panel, f"Live FPS: {int(fps)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        cv2.putText(panel, f"AI Limit: {int(FPS_TARGET)} FPS",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(panel, "OBJECT LIST",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        unique_labels = sorted(list(set(detected_labels)))

        for i, label in enumerate(unique_labels[:12]):
            cv2.putText(panel, f"- {label}",
                        (10, 160 + (i * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Alert Banner
        cv2.rectangle(frame, (0, 0), (w, 50), alert_color, -1)

        cv2.putText(frame, alert_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2)

        combined = np.hstack((frame, panel))

        cv2.imshow("YOLO Early FIRE/SMOKE detection and Alert System", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run Program
if __name__ == "__main__":
    main()
