👨‍💻 Face Recognition Attendance System

A Tkinter-based desktop application that uses OpenCV face recognition (LBPH) and SQLite to mark and manage attendance automatically.
This system captures faces, trains a recognition model, and marks attendance with timestamps, all through a modern GUI interface.

🚀 Features

Face Capture – Add new people with their face images (auto-saves dataset).

Model Training – Train an LBPH face recognizer on collected datasets.

Real-Time Recognition – Mark attendance automatically when a person is detected.

SQLite Database – Store attendance and person records.

Excel Export – Save attendance logs to .xlsx.

GUI (Tkinter) – Easy-to-use, modern interface with controls for all operations.

Thread-Safe Operations – Background threading for smooth video & DB operations.

🛠️ Tech Stack

OpenCV
 – Face detection & recognition

SQLite3
 – Database

Tkinter
 – GUI framework

Pandas
 – Exporting to Excel

Pillow
 – Image handling

📂 Project Structure
.
├── face_recognition_attendance.py   # Main application  
├── dataset/                         # Captured face images (auto-created)  
├── attendance.db                    # SQLite database  
├── face_recognizer.yml              # Trained LBPH model (auto-generated)  
├── label_map.pkl                    # Label ↔ Person name mapping  
└── README.md                        # Project documentation  


🎯 Usage

Add New Person → Enter name & capture 30 photos.

Train Model → Builds an LBPH model from the dataset.

Start Recognition → System detects faces via webcam and marks attendance.

View Attendance → Check records filtered by date.

Export to Excel → Save logs as .xlsx.

🔮 Future Improvements

Add deep learning-based recognition (e.g., FaceNet/Dlib).

Multi-camera support.

Admin login & role-based access.

Cloud database integration (Firebase/MySQL).
