# Fire and Smoke Detection System Using YOLO

A real-time computer vision prototype for detecting fire and smoke, analyzing visual hazard severity, identifying nearby human presence, and triggering multi-level alerts based on the detected situation.

The system combines two YOLO models with OpenCV-based fire validation, temporal verification, visual hazard growth analysis, human proximity estimation, and Twilio SMS notifications.

## Overview

Traditional fire detection systems generally rely on sensors such as smoke or temperature detectors. This project explores a computer vision-based approach that uses CCTV or camera footage to identify visible fire and smoke at an early stage.

The system uses two YOLO models:

- A general YOLOv8 model for detecting people and other objects in the scene.
- A custom-trained YOLO model for detecting fire, smoke, and flames.

To improve reliability, detected fire regions are additionally validated using HSV color analysis. The system also checks whether a hazard persists across multiple inference cycles before treating it as a verified event.

After verification, the prototype analyzes the approximate visual size and growth of the detected hazard along with the presence of nearby people. Based on these conditions, the system assigns an alert level and triggers the appropriate response.

## Key Features

- Real-time fire and smoke detection
- Dual YOLO model architecture
- General object and human detection using YOLOv8
- Custom fire and smoke detection model
- HSV-based validation of detected fire regions
- Temporal hazard verification to reduce transient false detections
- Visual hazard area estimation
- Hazard growth analysis across multiple inference cycles
- Human proximity estimation
- Three-level context-aware alert system
- Twilio SMS integration
- SMS cooldown and alert escalation handling
- Non-blocking SMS delivery using threading
- Real-time diagnostics panel
- Optional Twilio configuration

## System Architecture

The overall processing pipeline is:

```text
Camera / CCTV Feed
        |
        v
+-----------------------+
| General YOLOv8 Model  |
| Person/Object Detection|
+-----------------------+
        |
        |                 +-------------------------+
        +---------------->| Custom Fire/Smoke YOLO  |
                          +-------------------------+
                                      |
                                      v
                          +-------------------------+
                          | HSV Fire Validation     |
                          +-------------------------+
                                      |
                                      v
                          +-------------------------+
                          | Temporal Verification   |
                          +-------------------------+
                                      |
                                      v
                          +-------------------------+
                          | Hazard Area and Growth  |
                          | Analysis                |
                          +-------------------------+
                                      |
                                      v
                          +-------------------------+
                          | Human Proximity Check   |
                          +-------------------------+
                                      |
                                      v
                          +-------------------------+
                          | Alert Level Decision    |
                          +-------------------------+
                            |        |        |
                            v        v        v
                          Level 1  Level 2  Level 3
```

## Multi-Level Alert System

The prototype uses three alert levels based on the detected hazard and surrounding context.

### Level 1: Local Hazard

Triggered when a verified localized hazard is detected and a person is identified nearby.

Intended response:

- Trigger a local CCTV speaker, buzzer, or alarm system.
- Warn nearby people so that the situation can be addressed immediately.

The current software prototype represents the local alarm through the `trigger_local_alarm()` function. This function can later be connected to physical alarm hardware or an IoT-enabled CCTV system.

### Level 2: Unattended Hazard

Triggered when a verified hazard is detected without a nearby person available to respond.

Response:

- Trigger the local alarm.
- Send an SMS notification to the registered property owner using Twilio.

### Level 3: Critical Emergency

Triggered when the system detects a visually large hazard or rapid visual growth of the detected hazard.

Response:

- Trigger the local alarm.
- Notify the registered property owner.
- Send an additional alert to a configured emergency test contact.

For safety, the prototype does not contain or automatically contact real emergency-service numbers. During development and testing, personal test numbers should be used instead.

## How the Detection System Works

### 1. General Object Detection

The standard YOLOv8 model analyzes each frame and identifies people and other objects.

Person detections are used by the alert decision system to estimate whether someone is located near the detected hazard.

### 2. Fire and Smoke Detection

A custom YOLO model independently processes the same frame to detect:

- Fire
- Smoke
- Flames

A configurable confidence threshold is used to filter low-confidence detections.

### 3. HSV Fire Validation

Fire and flame detections undergo an additional computer vision validation step.

The detected region is extracted from the frame and converted from BGR to HSV color space. A predefined HSV range is then used to calculate the proportion of pixels that visually resemble fire-like colors.

The detection is accepted only when the calculated ratio exceeds the configured threshold.

This acts as an additional heuristic validation layer intended to reduce certain false-positive fire detections.

### 4. Temporal Hazard Verification

A single detection is not immediately treated as a confirmed hazard.

The system tracks whether valid hazards persist across multiple inference cycles. An alert is generated only after the configured verification streak is reached.

This helps reduce alerts caused by short-lived or unstable detections.

### 5. Hazard Area Estimation

The system calculates the bounding-box area of the primary detected hazard and normalizes it relative to the total camera frame area.

This provides an estimate of the hazard's visual size within the camera view.

This value should not be interpreted as the physical size or actual intensity of a fire because camera distance, viewing angle, and perspective can affect the apparent bounding-box area.

### 6. Hazard Growth Analysis

The normalized hazard area is stored across multiple inference cycles.

The system compares previous and current hazard areas to estimate whether the visible hazard is expanding rapidly.

This provides a simple visual growth indicator that can contribute to Level 3 escalation.

It is an image-based approximation and does not represent a direct physical measurement of fire-spread speed.

### 7. Human Proximity Analysis

The general YOLO model detects people in the scene.

The system calculates the distance between the center of a detected person's bounding box and the center of the primary hazard bounding box. The distance is normalized relative to the frame diagonal.

If the normalized distance falls below the configured threshold, the person is considered visually near the hazard.

This is a two-dimensional image-based proximity estimate rather than a measurement of real-world physical distance.

### 8. Alert Decision

After the hazard has been verified, the system evaluates:

- Visual hazard size
- Visual hazard growth
- Human proximity

These conditions are used to classify the situation into Level 1, Level 2, or Level 3.

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Twilio API
- Python Threading

## Project Structure

```text
fire-smoke-detection-using-YOLO/
|
|-- fire_detection_system.py    # Main detection and alert system
|-- keys_example.py             # Example Twilio configuration
|-- README.md                   # Project documentation
|-- .gitignore                  # Files excluded from version control
```

The trained model files are not included in the repository due to file-size limitations.

Required model files:

```text
yolov8n.pt
fire_model.pt
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fire-smoke-detection-using-YOLO.git
cd fire-smoke-detection-using-YOLO
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python numpy twilio
```

### 3. Add the Required Models

Place the following model files in the project directory:

```text
yolov8n.pt
fire_model.pt
```

The standard YOLOv8 model is used for general object and person detection.

The custom fire detection model is used for detecting fire, smoke, and flames.

## Twilio Configuration

Twilio integration is optional.

If Twilio is not configured, the detection system can continue running while SMS notifications remain disabled.

To enable SMS alerts, create a file named:

```text
keys.py
```

Use `keys_example.py` as a template:

```python
accountSID = "your_twilio_account_sid"
authToken = "your_twilio_auth_token"
twilioNumber = "your_twilio_virtual_number"

ownerNumber = "your_owner_test_number"
emergencyTestNumber = "your_emergency_test_number"
```

Never commit your actual `keys.py` file or Twilio credentials to a public GitHub repository.

Make sure `keys.py` is included in `.gitignore`.

## Running the Project

Run the main program:

```bash
python fire_detection_system.py
```

The system will:

1. Open the configured camera.
2. Run general object and person detection.
3. Run fire and smoke detection.
4. Validate detected fire regions.
5. Verify hazards across multiple inference cycles.
6. Analyze visual hazard area and growth.
7. Estimate whether a person is near the hazard.
8. Determine the appropriate alert level.
9. Trigger the corresponding local or SMS alert.

Press `Q` to stop the application.

## Real-Time Diagnostics

The application displays a live camera feed along with diagnostic information including:

- Current FPS
- Detected objects
- Hazard area ratio
- Estimated hazard growth
- Human proximity status
- Current alert level

Bounding boxes are displayed for detected people, fire, smoke, and ignored fire candidates.

## Dataset Sources

The project uses or references the following datasets:

### COCO Dataset

Used by the standard YOLOv8 model for general object and person detection.

### Fire Detection Dataset

A fire detection dataset was used for the custom fire and smoke detection component.

Dataset source: Kaggle Fire Detection Dataset by ironwolf437.

## Current Limitations

This project is an experimental prototype and is not intended to replace certified fire detection or emergency-response systems.

Current limitations include:

- Hazard growth is estimated from changes in visual bounding-box area rather than physical fire-spread measurements.
- Camera perspective and distance can affect visual hazard-size estimation.
- Human proximity is estimated in two-dimensional image space and does not represent actual physical distance.
- HSV validation uses predefined color ranges and may be affected by lighting conditions.
- Smoke detections rely primarily on the trained detection model.
- Local CCTV speaker or alarm hardware integration is represented through a software function and requires hardware-specific implementation.
- Emergency escalation uses test contacts during development rather than real emergency-service numbers.

## Future Improvements

Potential improvements include:

- Deploying the system on edge devices such as NVIDIA Jetson or Raspberry Pi-compatible hardware.
- Improving fire and smoke detection using larger and more diverse datasets.
- Implementing more robust temporal tracking of individual hazards.
- Using object tracking to improve person and hazard association.
- Improving fire-growth estimation using segmentation instead of bounding-box area.
- Adding depth estimation for better human-to-hazard proximity analysis.
- Integrating physical alarms, CCTV speakers, and IoT devices.
- Developing a cloud-based monitoring dashboard.
- Recording hazard events and alert history.
- Adding camera location information to alerts.
- Supporting multiple CCTV camera feeds.
- Implementing secure configuration using environment variables.

## Safety Notice

This project was developed as an educational and experimental prototype.

It should not be used as the sole fire detection, emergency notification, or life-safety system in real-world environments.

Automated contact with real emergency services should only be implemented in an appropriately tested and authorized production system. Personal or designated test numbers should be used during development.

## Author

M. Sree Ravindranath

B.Tech Electronics and Communication Engineering  
Santhiram Engineering College, Nandyal
