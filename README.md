# 🔥 Fire & Smoke Detection System using YOLO

## 📌 Overview
This project implements a real-time fire and smoke detection system using YOLO and computer vision techniques. It detects fire hazards, validates them using HSV color filtering, and sends alerts using Twilio.

---

## 🚀 Features
- Real-time fire and smoke detection
- Dual YOLO model (object + fire)
- HSV-based fire validation
- Temporal hazard verification
- Context-aware alert system
- SMS alerts using Twilio

---

## 🛠️ Technologies Used
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- Twilio API

---

## 📂 Project Structurefire_detection_system.py 
# Main program keys_example.py 
# Example credentials file README.md       
# Project documentation .gitignore          
# Ignored files

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/fire-smoke-detection-using-YOLO.git
cd fire-smoke-detection-using-YOLO
2️⃣ Install Dependencies
Bash
pip install ultralytics opencv-python numpy twilio
▶️ How to Run
Create a file named keys.py and add your Twilio credentials:
Python
accountSID = "your_account_sid"
authToken = "your_auth_token"
twilioNumber = "your_twilio_number"
targetNumber = "your_phone_number"
Place model files in the project directory:
yolov8n.pt
fire_model.pt
Run the program:
Bash
python fire_detection_system.py
⚠️ Important Notes
Model files are not included due to GitHub size limitations
Do NOT upload your credentials file (keys.py) to GitHub
Use keys_example.py as a reference
📊 System Workflow
Capture video frames
Detect objects and humans using YOLO
Detect fire and smoke using custom model
Validate fire using HSV color filtering
Apply temporal verification
Perform context-aware analysis
Trigger alerts and send SMS
📚 Dataset Sources
COCO Dataset: https://cocodataset.org⁠
Fire Dataset (Kaggle): (https://www.kaggle.com/datasets/ironwolf437/fire-detection-dataset)
🔮 Future Improvements
Deploy on edge devices (Raspberry Pi, Jetson Nano)
Improve fire classification accuracy using larger datasets
Integrate IoT-based fire suppression systems
Add cloud monitoring dashboard
👨‍💻 Author
M.Sree Ravindranath
⭐ Support
If you found this project useful, consider giving it a ⭐ on GitHub!
