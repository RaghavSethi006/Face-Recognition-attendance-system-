import cv2
import numpy as np
import os
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, date
import pandas as pd
from PIL import Image, ImageTk
import threading

class FaceRecognitionAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.dataset_path = "dataset"
        self.cap = None
        self.is_training = False
        self.is_recognizing = False
        self.label_map = {}
        self.current_person_name = ""
        self.photo_count = 0
        
        # Create dataset directory if it doesn't exist
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        # Initialize database
        self.init_database()
        
        # Create GUI
        self.create_gui()
        
        # Load existing model if available
        self.load_model()
    
    def init_database(self):
        """Initialize SQLite database for attendance records"""
        self.conn = sqlite3.connect('attendance.db')
        self.cursor = self.conn.cursor()
        
        # Create attendance table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT DEFAULT 'Present'
            )
        ''')
        
        # Create persons table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                added_date TEXT NOT NULL
            )
        ''')
        
        self.conn.commit()
    
    def create_gui(self):
        """Create the main GUI interface"""
        # Title
        title_label = tk.Label(self.root, text="Face Recognition Attendance System", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left frame for controls
        left_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Control buttons
        control_label = tk.Label(left_frame, text="Controls", font=('Arial', 14, 'bold'), 
                                fg='white', bg='#34495e')
        control_label.pack(pady=10)
        
        # Add new person section
        add_person_frame = tk.LabelFrame(left_frame, text="Add New Person", 
                                        font=('Arial', 10, 'bold'), fg='white', bg='#34495e')
        add_person_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(add_person_frame, text="Name:", fg='white', bg='#34495e').pack(anchor='w')
        self.name_entry = tk.Entry(add_person_frame, font=('Arial', 10))
        self.name_entry.pack(fill=tk.X, pady=2)
        
        self.capture_btn = tk.Button(add_person_frame, text="Capture Photos", 
                                    command=self.start_photo_capture, bg='#3498db', fg='white',
                                    font=('Arial', 10, 'bold'))
        self.capture_btn.pack(fill=tk.X, pady=2)
        
        self.photo_count_label = tk.Label(add_person_frame, text="Photos captured: 0", 
                                         fg='white', bg='#34495e')
        self.photo_count_label.pack(pady=2)
        
        # Training section
        train_frame = tk.LabelFrame(left_frame, text="Model Training", 
                                   font=('Arial', 10, 'bold'), fg='white', bg='#34495e')
        train_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.train_btn = tk.Button(train_frame, text="Train Model", 
                                  command=self.train_model, bg='#e67e22', fg='white',
                                  font=('Arial', 10, 'bold'))
        self.train_btn.pack(fill=tk.X, pady=2)
        
        self.train_status = tk.Label(train_frame, text="Status: Ready", 
                                    fg='white', bg='#34495e')
        self.train_status.pack(pady=2)
        
        # Attendance section
        attendance_frame = tk.LabelFrame(left_frame, text="Attendance", 
                                        font=('Arial', 10, 'bold'), fg='white', bg='#34495e')
        attendance_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.recognize_btn = tk.Button(attendance_frame, text="Start Recognition", 
                                      command=self.toggle_recognition, bg='#27ae60', fg='white',
                                      font=('Arial', 10, 'bold'))
        self.recognize_btn.pack(fill=tk.X, pady=2)
        
        self.view_attendance_btn = tk.Button(attendance_frame, text="View Attendance", 
                                            command=self.view_attendance, bg='#8e44ad', fg='white',
                                            font=('Arial', 10, 'bold'))
        self.view_attendance_btn.pack(fill=tk.X, pady=2)
        
        self.export_btn = tk.Button(attendance_frame, text="Export to Excel", 
                                   command=self.export_attendance, bg='#16a085', fg='white',
                                   font=('Arial', 10, 'bold'))
        self.export_btn.pack(fill=tk.X, pady=2)
        
        # Right frame for video display
        right_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video display
        self.video_label = tk.Label(right_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, 
                                  anchor='w', bg='#ecf0f1')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def start_photo_capture(self):
        """Start capturing photos for a new person"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name first!")
            return
        
        self.current_person_name = name
        self.photo_count = 0
        
        # Create directory for the person
        person_dir = os.path.join(self.dataset_path, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Add person to database (thread-safe)
        def add_person_to_db():
            try:
                conn = sqlite3.connect('attendance.db')
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO persons (name, added_date) VALUES (?, ?)", 
                               (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                self.root.after(0, lambda: messagebox.showerror("Database Error", str(e)))
        
        threading.Thread(target=add_person_to_db, daemon=True).start()
        
        self.capture_photos()
    
    def capture_photos(self):
        """Capture photos for training"""
        self.cap = cv2.VideoCapture(0)
        
        def capture_loop():
            target_photos = 30
            while self.photo_count < target_photos and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Save face image
                    face = gray[y:y + h, x:x + w]
                    person_dir = os.path.join(self.dataset_path, self.current_person_name)
                    filename = f"{self.current_person_name}_{self.photo_count + 1}.jpg"
                    cv2.imwrite(os.path.join(person_dir, filename), face)
                    self.photo_count += 1
                    
                    # Update GUI
                    self.root.after(0, lambda: self.photo_count_label.config(
                        text=f"Photos captured: {self.photo_count}/{target_photos}"))
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((640, 480))
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.root.after(0, lambda img=frame_tk: self.video_label.config(image=img))
                self.root.after(0, lambda img=frame_tk: setattr(self.video_label, 'image', img))
                
                cv2.waitKey(100)  # Delay between captures
            
            self.cap.release()
            self.root.after(0, lambda: self.status_bar.config(
                text=f"Photo capture completed for {self.current_person_name}"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", f"Captured {self.photo_count} photos for {self.current_person_name}"))
        
        threading.Thread(target=capture_loop, daemon=True).start()
    
    def train_model(self):
        """Train the face recognition model"""
        self.train_status.config(text="Status: Training...")
        self.train_btn.config(state='disabled')
        
        def train_thread():
            try:
                faces, labels, label_map = self.prepare_training_data()
                if len(faces) == 0:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "No training data found. Please add some people first."))
                    return
                
                self.recognizer.train(faces, np.array(labels))
                self.recognizer.save('face_recognizer.yml')
                self.label_map = label_map
                
                # Save label map
                import pickle
                with open('label_map.pkl', 'wb') as f:
                    pickle.dump(label_map, f)
                
                self.root.after(0, lambda: self.train_status.config(text="Status: Training Complete"))
                self.root.after(0, lambda: self.status_bar.config(text="Model training completed"))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Model training completed!"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            finally:
                self.root.after(0, lambda: self.train_btn.config(state='normal'))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def prepare_training_data(self):
        """Prepare training data from dataset"""
        faces = []
        labels = []
        label_map = {}
        label_id = 0
        
        for person_name in os.listdir(self.dataset_path):
            person_dir = os.path.join(self.dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            label_map[label_id] = person_name
            
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    faces.append(image)
                    labels.append(label_id)
            
            label_id += 1
        
        return faces, labels, label_map
    
    def load_model(self):
        """Load existing trained model"""
        try:
            if os.path.exists('face_recognizer.yml'):
                self.recognizer.read('face_recognizer.yml')
                import pickle
                with open('label_map.pkl', 'rb') as f:
                    self.label_map = pickle.load(f)
                self.train_status.config(text="Status: Model Loaded")
        except Exception as e:
            print(f"Could not load existing model: {e}")
    
    def toggle_recognition(self):
        """Start or stop face recognition"""
        if not self.is_recognizing:
            self.start_recognition()
        else:
            self.stop_recognition()
    
    def start_recognition(self):
        """Start face recognition for attendance"""
        if not os.path.exists('face_recognizer.yml'):
            messagebox.showerror("Error", "No trained model found. Please train the model first.")
            return
        
        self.is_recognizing = True
        self.recognize_btn.config(text="Stop Recognition", bg='#e74c3c')
        self.cap = cv2.VideoCapture(0)
        
        def recognition_loop():
            while self.is_recognizing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    label, confidence = self.recognizer.predict(face)
                    
                    if confidence < 50:  # Confidence threshold
                        name = self.label_map.get(label, "Unknown")
                        color = (0, 255, 0)  # Green for recognized
                        self.mark_attendance(name)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Red for unknown
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Convert and display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((640, 480))
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.root.after(0, lambda img=frame_tk: self.video_label.config(image=img))
                self.root.after(0, lambda img=frame_tk: setattr(self.video_label, 'image', img))
                
                cv2.waitKey(30)
            
            if self.cap:
                self.cap.release()
        
        threading.Thread(target=recognition_loop, daemon=True).start()
    
    def stop_recognition(self):
        """Stop face recognition"""
        self.is_recognizing = False
        self.recognize_btn.config(text="Start Recognition", bg='#27ae60')
        if self.cap:
            self.cap.release()
        self.status_bar.config(text="Recognition stopped")
    
    def mark_attendance(self, name):
        """Mark attendance for a person"""
        if name == "Unknown":
            return
        
        today = date.today().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Use thread-safe database operations
        def db_operation():
            try:
                # Create new connection for this thread
                conn = sqlite3.connect('attendance.db')
                cursor = conn.cursor()
                
                # Check if attendance already marked today
                cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, today))
                if cursor.fetchone() is None:
                    cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                                   (name, today, current_time))
                    conn.commit()
                    self.root.after(0, lambda: self.status_bar.config(
                        text=f"Attendance marked for {name} at {current_time}"))
                
                conn.close()
            except sqlite3.Error as e:
                print(f"Database error: {e}")
        
        threading.Thread(target=db_operation, daemon=True).start()
    
    def view_attendance(self):
        """Open attendance viewer window"""
        attendance_window = tk.Toplevel(self.root)
        attendance_window.title("Attendance Records")
        attendance_window.geometry("800x600")
        attendance_window.configure(bg='#2c3e50')
        
        # Date filter frame
        filter_frame = tk.Frame(attendance_window, bg='#34495e')
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filter by Date:", fg='white', bg='#34495e').pack(side=tk.LEFT)
        date_var = tk.StringVar(value=date.today().strftime("%Y-%m-%d"))
        date_entry = tk.Entry(filter_frame, textvariable=date_var)
        date_entry.pack(side=tk.LEFT, padx=5)
        
        def refresh_data():
            # Clear existing data
            for item in tree.get_children():
                tree.delete(item)
            
            # Fetch filtered data using thread-safe connection
            try:
                conn = sqlite3.connect('attendance.db')
                cursor = conn.cursor()
                
                selected_date = date_var.get()
                if selected_date:
                    cursor.execute("SELECT * FROM attendance WHERE date = ? ORDER BY time", (selected_date,))
                else:
                    cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time")
                
                records = cursor.fetchall()
                for record in records:
                    tree.insert("", tk.END, values=record[1:])  # Skip ID
                
                conn.close()
            except sqlite3.Error as e:
                messagebox.showerror("Database Error", str(e))
        
        filter_btn = tk.Button(filter_frame, text="Filter", command=refresh_data, bg='#3498db', fg='white')
        filter_btn.pack(side=tk.LEFT, padx=5)
        
        show_all_btn = tk.Button(filter_frame, text="Show All", 
                                command=lambda: (date_var.set(""), refresh_data()), bg='#e67e22', fg='white')
        show_all_btn.pack(side=tk.LEFT, padx=5)
        
        # Treeview for attendance records
        tree_frame = tk.Frame(attendance_window, bg='#2c3e50')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ("Name", "Date", "Time", "Status")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load initial data
        refresh_data()
    
    def export_attendance(self):
        """Export attendance to Excel file"""
        try:
            # Use thread-safe database connection
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name, date, time, status FROM attendance ORDER BY date DESC, time")
            records = cursor.fetchall()
            conn.close()
            
            if not records:
                messagebox.showinfo("Info", "No attendance records to export.")
                return
            
            df = pd.DataFrame(records, columns=['Name', 'Date', 'Time', 'Status'])
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if filename:
                df.to_excel(filename, index=False)
                messagebox.showinfo("Success", f"Attendance exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

def main():
    root = tk.Tk()
    app = FaceRecognitionAttendanceSystem(root)
    
    def on_closing():
        if app.cap:
            app.cap.release()
        if hasattr(app, 'conn'):
            app.conn.close()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()