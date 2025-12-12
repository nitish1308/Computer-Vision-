# Face Recognition Attendance System  
A real-time attendance monitoring application built using **Python**, **OpenCV**, and **Streamlit**, allowing user registration, model training, and attendance marking via webcam.

---

## 📌 Overview
The **Face Recognition Attendance System** enables accurate, automated attendance tracking using facial recognition. Users can register themselves through the webcam, after which the system trains a recognition model. Attendance is marked when a registered user’s face is identified. All functionality is provided through an intuitive Streamlit interface.

---

## 🚀 Features

### ✔ User Registration  
- Capture live images from webcam  
- Automatic face detection and cropping  
- Save multiple samples per user  
- Organized dataset structure:



### ✔ Model Training  
- Uses **LBPH (Local Binary Pattern Histogram)** recognizer  
- Generates:
- `trainer.yml` (trained model)
- `labels.pickle` (mapping of user labels)

### ✔ Attendance Marking  
- Captures webcam frame  
- Recognizes the face  
- Logs attendance in `attendance.csv` with:
- Date  
- Time  
- User Name  

### ✔ Streamlit UI  
Tabbed interface for easy navigation:

### ✔ Dataset & Model Management  
- Browse dataset images  
- Track total samples  
- Clean modular structure  

---

## 📂 Project Structure

face-recognition-attendance-system/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── dataset/                   # User image folders (auto-created)
│     ├── user_1_John/
│     ├── user_2_Alice/
│     └── ...
├── trainer.yml                # LBPH model (generated after training)
├── labels.pickle              # User label mappings
├── attendance.csv             # Attendance logs
└── README.md                  # Detailed documentation


Webcam → Frame Capture → Face Detection → Face Cropping 
        ↓
 User Registration → Save Dataset → Train Model (LBPH)
        ↓
 Attendance Marking → Face Recognition → Log to CSV → Dashboard View
