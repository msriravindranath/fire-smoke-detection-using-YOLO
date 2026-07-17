import cv2
import os
import time
import threading
from collections import deque

import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Model paths
OBJ_MODEL_PATH = "yolov8n.pt"
FIRE_MODEL_PATH = "fire_model.pt"

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection
FIRE_CONFIDENCE = 0.65
REQUIRED_STREAK = 3

# Fire HSV validation
HSV_FIRE_RATIO_THRESHOLD = 0.02

# Hazard size thresholds
# These are normalized relative to the total frame area.
SMALL_HAZARD_RATIO = 0.02
LARGE_HAZARD_RATIO = 0.10

# Person proximity
# Distance is normalized by the frame diagonal.
PERSON_NEAR_DISTANCE_RATIO = 0.25

# Fire growth tracking
GROWTH_HISTORY_SIZE = 8
RAPID_GROWTH_THRESHOLD = 0.04

# Alert cooldown
SMS_COOLDOWN = 60

# Performance
FPS_TARGET = 15.0
INFERENCE_INTERVAL = 1.0 / FPS_TARGET


# ============================================================
# OPTIONAL TWILIO CONFIGURATION
# ============================================================

try:
    import keys
    from twilio.rest import Client

    TWILIO_ENABLED = True
    print("Twilio configuration loaded.")

except ImportError:
    TWILIO_ENABLED = False
    Client = None

    print(
        "WARNING: Twilio is not configured. "
        "Detection will continue without SMS alerts."
    )


# ============================================================
# LOAD MODELS
# ============================================================

if not os.path.exists(FIRE_MODEL_PATH):
    raise FileNotFoundError(
        f"Fire detection model not found: {FIRE_MODEL_PATH}"
    )

try:
    model_obj = YOLO(OBJ_MODEL_PATH)
    model_fire = YOLO(FIRE_MODEL_PATH)

except Exception as e:
    raise RuntimeError(
        f"Failed to load YOLO models: {e}"
    )


# ============================================================
# GLOBAL SYSTEM STATE
# ============================================================

current_streak = 0

last_sms_time = 0
last_alert_level = 0

fire_area_history = deque(
    maxlen=GROWTH_HISTORY_SIZE
)


# ============================================================
# ALERT SYSTEM
# ============================================================

def get_recipient(level):
    """
    Return the configured recipient for the alert level.

    Level 2:
        Property owner.

    Level 3:
        Emergency TEST contact.

    Real emergency-service numbers should NOT be used during
    prototype testing.
    """

    if level == 2:
        return getattr(
            keys,
            "ownerNumber",
            getattr(keys, "targetNumber", None)
        )

    if level == 3:
        return getattr(
            keys,
            "emergencyTestNumber",
            getattr(keys, "targetNumber", None)
        )

    return None


def trigger_local_alarm(message):
    """
    Prototype local alarm.

    Replace this function with actual CCTV speaker,
    buzzer, GPIO, or IoT alarm integration during
    hardware deployment.
    """

    print(f"\nLOCAL ALARM: {message}")


def trigger_alert(message, level):
    """
    Execute the appropriate response according to
    the hazard level.

    Level 1:
        Local alarm only.

    Level 2:
        Local alarm + owner SMS.

    Level 3:
        Local alarm + owner SMS +
        emergency TEST contact SMS.
    """

    global last_sms_time
    global last_alert_level

    # Local alarm is triggered at every verified level.
    trigger_local_alarm(message)

    # Level 1 requires only local response.
    if level == 1:
        last_alert_level = max(last_alert_level, level)
        return

    if not TWILIO_ENABLED:
        print(
            f"SMS disabled. Simulated Level {level} alert."
        )

        last_alert_level = max(
            last_alert_level,
            level
        )

        return

    now = time.time()

    is_escalation = level > last_alert_level

    cooldown_expired = (
        now - last_sms_time >= SMS_COOLDOWN
    )

    if not (
        cooldown_expired or is_escalation
    ):
        return

    last_sms_time = now
    last_alert_level = level

    recipients = []

    # Owner should receive both Level 2 and Level 3.
    owner = getattr(
        keys,
        "ownerNumber",
        getattr(keys, "targetNumber", None)
    )

    if owner:
        recipients.append(owner)

    # Level 3 additionally alerts the emergency test contact.
    if level == 3:

        emergency_test = getattr(
            keys,
            "emergencyTestNumber",
            None
        )

        if (
            emergency_test
            and emergency_test not in recipients
        ):
            recipients.append(
                emergency_test
            )

    for recipient in recipients:

        threading.Thread(
            target=send_sms,
            args=(
                message,
                recipient,
                level
            ),
            daemon=True
        ).start()


def send_sms(
    message,
    recipient,
    level
):

    try:

        client = Client(
            keys.accountSID,
            keys.authToken
        )

        client.messages.create(

            body=(
                "YOLO Fire/Smoke Detection System\n"
                f"Alert Level: {level}\n"
                f"{message}"
            ),

            from_=keys.twilioNumber,

            to=recipient
        )

        print(
            f"SMS sent successfully "
            f"for Level {level}."
        )

    except Exception as e:

        print(
            f"SMS failed: {e}"
        )


# ============================================================
# FIRE VALIDATION
# ============================================================

def is_physically_fire(
    frame,
    box
):

    x1, y1, x2, y2 = map(
        int,
        box
    )

    height, width = frame.shape[:2]

    x1 = max(
        0,
        min(x1, width)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y1 = max(
        0,
        min(y1, height)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return False

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    lower_fire = np.array(
        [0, 100, 200]
    )

    upper_fire = np.array(
        [35, 255, 255]
    )

    mask = cv2.inRange(
        hsv,
        lower_fire,
        upper_fire
    )

    total_pixels = (
        roi.shape[0]
        * roi.shape[1]
    )

    if total_pixels == 0:
        return False

    fire_pixels = cv2.countNonZero(
        mask
    )

    fire_ratio = (
        fire_pixels
        / total_pixels
    )

    return (
        fire_ratio
        > HSV_FIRE_RATIO_THRESHOLD
    )


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def box_center(box):

    x1, y1, x2, y2 = box

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    return (
        center_x,
        center_y
    )


def person_is_near_hazard(
    person_box,
    hazard_box,
    frame_width,
    frame_height
):

    person_x, person_y = box_center(
        person_box
    )

    hazard_x, hazard_y = box_center(
        hazard_box
    )

    distance = np.sqrt(

        (person_x - hazard_x) ** 2

        +

        (person_y - hazard_y) ** 2

    )

    frame_diagonal = np.sqrt(

        frame_width ** 2

        +

        frame_height ** 2

    )

    normalized_distance = (
        distance
        / frame_diagonal
    )

    return (
        normalized_distance
        <= PERSON_NEAR_DISTANCE_RATIO
    )


# ============================================================
# FIRE GROWTH ANALYSIS
# ============================================================

def calculate_growth_rate():

    if len(
        fire_area_history
    ) < 2:

        return 0.0

    oldest_area = (
        fire_area_history[0]
    )

    newest_area = (
        fire_area_history[-1]
    )

    return (
        newest_area
        - oldest_area
    )


# ============================================================
# MAIN SYSTEM
# ============================================================

def main():

    global current_streak
    global last_alert_level

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open camera."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    prev_frame_time = 0
    last_inference_time = 0

    object_boxes = []
    person_boxes = []

    fire_boxes = []
    smoke_boxes = []
    ignored_boxes = []

    detected_labels = []

    alert_text = "SYSTEM SAFE"

    alert_color = (
        0,
        255,
        0
    )

    print(
        "\nFire Detection System Started"
    )

    print(
        "Press Q to quit.\n"
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Failed to read camera frame."
            )

            break

        current_time = time.time()

        if prev_frame_time > 0:

            fps = (

                1

                /

                (
                    current_time
                    - prev_frame_time
                )

            )

        else:

            fps = 0

        prev_frame_time = (
            current_time
        )

        if (

            current_time
            - last_inference_time

            >= INFERENCE_INTERVAL

        ):

            last_inference_time = (
                current_time
            )

            object_boxes.clear()
            person_boxes.clear()

            fire_boxes.clear()
            smoke_boxes.clear()
            ignored_boxes.clear()

            detected_labels.clear()

            frame_height, frame_width = (
                frame.shape[:2]
            )

            frame_area = (

                frame_width

                *

                frame_height

            )

            max_hazard_area = 0

            primary_hazard_box = None

            person_near = False

            # =================================================
            # GENERAL OBJECT DETECTION
            # =================================================

            obj_results = model_obj(
                frame,
                verbose=False
            )[0]

            for box in obj_results.boxes:

                cls = int(
                    box.cls[0]
                )

                label = (
                    model_obj.names[cls]
                )

                detected_labels.append(
                    label
                )

                x1, y1, x2, y2 = map(

                    int,

                    box.xyxy[0]

                )

                current_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                # COCO class 0 = person
                if cls == 0:

                    person_boxes.append(
                        current_box
                    )

                else:

                    object_boxes.append(
                        current_box
                    )


            # =================================================
            # FIRE / SMOKE DETECTION
            # =================================================

            fire_results = model_fire(

                frame,

                verbose=False,

                conf=FIRE_CONFIDENCE

            )[0]

            valid_hazard_found = False

            for box in fire_results.boxes:

                cls = int(
                    box.cls[0]
                )

                label = (

                    model_fire
                    .names[cls]
                    .lower()

                )

                coords = (
                    box.xyxy[0]
                )

                x1, y1, x2, y2 = map(

                    int,

                    coords

                )

                hazard_box = (

                    x1,
                    y1,
                    x2,
                    y2

                )

                if label not in [

                    "fire",
                    "smoke",
                    "flame"

                ]:

                    continue


                # Smoke is accepted from the trained model.
                if label == "smoke":

                    is_valid = True

                    smoke_boxes.append(
                        hazard_box
                    )

                else:

                    is_valid = (
                        is_physically_fire(
                            frame,
                            coords
                        )
                    )

                    if is_valid:

                        fire_boxes.append(
                            hazard_box
                        )

                    else:

                        ignored_boxes.append(
                            hazard_box
                        )


                if is_valid:

                    valid_hazard_found = True

                    area = max(

                        0,

                        (
                            x2 - x1
                        )

                        *

                        (
                            y2 - y1
                        )

                    )

                    if (
                        area
                        > max_hazard_area
                    ):

                        max_hazard_area = (
                            area
                        )

                        primary_hazard_box = (
                            hazard_box
                        )


            # =================================================
            # TEMPORAL VERIFICATION
            # =================================================

            if valid_hazard_found:

                current_streak += 1

                current_streak = min(

                    current_streak,

                    REQUIRED_STREAK + 5

                )

            else:

                current_streak = max(

                    0,

                    current_streak - 1

                )

                if current_streak == 0:

                    last_alert_level = 0

                    fire_area_history.clear()


            # =================================================
            # NORMALIZED HAZARD AREA
            # =================================================

            hazard_area_ratio = (

                max_hazard_area
                / frame_area

                if frame_area > 0

                else 0

            )

            if valid_hazard_found:

                fire_area_history.append(
                    hazard_area_ratio
                )


            growth_rate = (
                calculate_growth_rate()
            )


            # =================================================
            # PERSON PROXIMITY ANALYSIS
            # =================================================

            if primary_hazard_box:

                for person_box in person_boxes:

                    if person_is_near_hazard(

                        person_box,

                        primary_hazard_box,

                        frame_width,

                        frame_height

                    ):

                        person_near = True

                        break


            # =================================================
            # MULTI-LEVEL ALERT LOGIC
            # =================================================

            alert_text = (
                "SYSTEM SAFE"
            )

            alert_color = (
                0,
                255,
                0
            )


            if (

                current_streak

                >= REQUIRED_STREAK

            ):

                rapid_growth = (

                    growth_rate

                    >= RAPID_GROWTH_THRESHOLD

                )

                large_hazard = (

                    hazard_area_ratio

                    >= LARGE_HAZARD_RATIO

                )


                # LEVEL 3
                #
                # Rapidly expanding hazard
                # OR visually large hazard.
                #
                # Uses TEST emergency contact only.
                if (

                    rapid_growth

                    or large_hazard

                ):

                    alert_text = (

                        "LEVEL 3: "
                        "CRITICAL EMERGENCY"

                    )

                    alert_color = (
                        0,
                        0,
                        255
                    )

                    trigger_alert(

                        (
                            "Critical hazard detected. "
                            "Rapid fire growth or large "
                            "hazard area observed. "
                            "Immediate action required."
                        ),

                        3

                    )


                # LEVEL 1
                #
                # Small/localized hazard
                # with a nearby person.
                elif person_near:

                    alert_text = (

                        "LEVEL 1: "
                        "LOCAL HAZARD"

                    )

                    alert_color = (
                        0,
                        255,
                        255
                    )

                    trigger_alert(

                        (
                            "Localized hazard detected "
                            "with human presence nearby."
                        ),

                        1

                    )


                # LEVEL 2
                #
                # Verified hazard with no nearby person.
                else:

                    alert_text = (

                        "LEVEL 2: "
                        "UNATTENDED HAZARD"

                    )

                    alert_color = (
                        0,
                        165,
                        255
                    )

                    trigger_alert(

                        (
                            "Verified unattended hazard "
                            "detected. Property owner "
                            "notification required."
                        ),

                        2

                    )


            elif current_streak > 0:

                alert_text = (

                    "VERIFYING HAZARD "

                    f"({current_streak}/"
                    f"{REQUIRED_STREAK})"

                )

                alert_color = (
                    0,
                    100,
                    255
                )


        # =====================================================
        # DRAW DETECTIONS
        # =====================================================

        for (
            x1,
            y1,
            x2,
            y2
        ) in object_boxes:

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                1

            )


        for (
            x1,
            y1,
            x2,
            y2
        ) in person_boxes:

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (255, 0, 0),

                2

            )

            cv2.putText(

                frame,

                "Person",

                (
                    x1,
                    max(20, y1 - 5)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 0, 0),

                1

            )


        for (
            x1,
            y1,
            x2,
            y2
        ) in fire_boxes:

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 0, 255),

                3

            )

            cv2.putText(

                frame,

                "CONFIRMED FIRE",

                (
                    x1,
                    max(20, y1 - 10)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2

            )


        for (
            x1,
            y1,
            x2,
            y2
        ) in smoke_boxes:

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (150, 150, 150),

                3

            )

            cv2.putText(

                frame,

                "SMOKE DETECTED",

                (
                    x1,
                    max(20, y1 - 10)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (150, 150, 150),

                2

            )


        for (
            x1,
            y1,
            x2,
            y2
        ) in ignored_boxes:

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (100, 100, 100),

                2

            )

            cv2.putText(

                frame,

                "IGNORED",

                (
                    x1,
                    max(20, y1 - 10)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (100, 100, 100),

                2

            )


        # =====================================================
        # DIAGNOSTICS PANEL
        # =====================================================

        height, width = (
            frame.shape[:2]
        )

        panel = np.zeros(

            (
                height,
                250,
                3
            ),

            dtype=np.uint8

        )


        cv2.putText(

            panel,

            "DIAGNOSTICS",

            (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (200, 200, 200),

            2

        )


        cv2.putText(

            panel,

            f"Live FPS: {int(fps)}",

            (10, 65),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            1

        )


        cv2.putText(

            panel,

            (
                "Hazard Area: "
                f"{hazard_area_ratio:.3f}"
            ),

            (10, 95),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            1

        )


        cv2.putText(

            panel,

            (
                "Growth: "
                f"{growth_rate:.3f}"
            ),

            (10, 125),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            1

        )


        cv2.putText(

            panel,

            (
                "Person Near: "
                f"{person_near}"
            ),

            (10, 155),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            1

        )


        cv2.putText(

            panel,

            "OBJECT LIST",

            (10, 195),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (200, 200, 200),

            2

        )


        unique_labels = sorted(

            set(
                detected_labels
            )

        )


        for i, label in enumerate(

            unique_labels[:10]

        ):

            cv2.putText(

                panel,

                f"- {label}",

                (
                    10,
                    225 + i * 22
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 255, 255),

                1

            )


        # =====================================================
        # ALERT BANNER
        # =====================================================

        cv2.rectangle(

            frame,

            (0, 0),

            (width, 50),

            alert_color,

            -1

        )


        cv2.putText(

            frame,

            alert_text,

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (255, 255, 255),

            2

        )


        combined = np.hstack(

            (
                frame,
                panel
            )

        )


        cv2.imshow(

            (
                "YOLO Fire/Smoke Detection "
                "and Alert System"
            ),

            combined

        )


        if (

            cv2.waitKey(1)

            & 0xFF

            == ord("q")

        ):

            break


    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
