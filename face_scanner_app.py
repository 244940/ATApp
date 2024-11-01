import cv2
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
from database_manager import DatabaseManager
import face_recognition
from datetime import datetime

class FaceScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Scanner")
        
        self.root.attributes('-fullscreen', False)
        
        # Set the background color
        self.root.configure(bg='#F0F0F0')
        
        # Create a frame to hold all widgets
        self.main_frame = tk.Frame(root, bg='#F0F0F0')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a label to display the video stream
        self.video_label = Label(self.main_frame, bg='#F0F0F0') 
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Create labels for date and time
        self.date_label = Label(self.main_frame, font=("Arial", 14), bg='#F0F0F0')
        self.date_label.pack(side=tk.TOP, pady=10)
        
        self.time_label = Label(self.main_frame, font=("Arial", 14), bg='#F0F0F0')  
        self.time_label.pack(side=tk.TOP, pady=10)
        
        # Create a label to display scanning result
        self.result_label = Label(self.main_frame, text="", font=("Arial", 16), bg='#F0F0F0') 
        self.result_label.pack(side=tk.BOTTOM, pady=10)
        
        # Initialize database manager
        try:
            self.db_manager = DatabaseManager()
            # Load known faces from database
            self.known_face_encodings, self.known_face_names, self.known_face_ids = self.db_manager.load_known_faces()
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to connect to the database: {str(e)}")
            self.db_manager = None
            self.known_face_encodings, self.known_face_names, self.known_face_ids = [], [], []
        
        #set up video capture
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
        except Exception as e:
            messagebox.showwarning("Camera Error", f"Failed to initialize camera: {str(e)}\nRunning without live video.")
            self.cap = None
        
        # Start the video loop and update time/date
        self.update_frame()
        self.update_datetime()
        
        # Set closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Escape key to close the application
        self.root.bind('<Escape>', lambda e: self.on_closing())
        
        # Bind resizing event to adjust text size dynamically
        self.root.bind('<Configure>', self.adjust_text_size)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process frame (face detection and recognition)
                self.process_frame(frame)
                
                # Convert the frame to PhotoImage
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            else:
                
                self.video_label.config(text="No camera feed available")
        
        # Call this function again after 10 milliseconds
        self.root.after(4, self.update_frame)

    def update_datetime(self):
        now = datetime.now()
        self.date_label.config(text=now.strftime("%Y-%m-%d"))
        self.time_label.config(text=now.strftime("%H:%M:%S"))
        self.root.after(1000, self.update_datetime)  # Update every second

    def adjust_text_size(self, event=None):
        
        window_width = self.root.winfo_width()
        font_size = max(14, window_width // 50)  
        
        self.date_label.config(font=("Arial", font_size))
        self.time_label.config(font=("Arial", font_size))

    def process_frame(self, frame):
        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            user_id = None
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                user_id = self.known_face_ids[first_match_index]
            
            # Attendance status
            attendance_text = "No attendance record"
            if user_id is not None and self.db_manager:
                current_schedule = self.db_manager.get_current_schedule(user_id)
                if current_schedule:
                    schedule_id, start_time, end_time, course_name = current_schedule
                    status = self.db_manager.log_attendance(user_id, schedule_id)
                    if status == "Too soon to log again":
                        attendance_text = f"Attendance recently logged for {course_name}"
                    else:
                        attendance_text = f"Attendance logged for {course_name}"
                else:
                    attendance_text = f"No class scheduled"
            
            # Draw a box around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw a label with name above the face
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 10), font, 1.0, (255, 255, 255), 1) 
            
            # Attendance text
            (text_width, text_height), _ = cv2.getTextSize(attendance_text, font, 0.8, 1)
            if text_width > (right - left):
                attendance_lines = [attendance_text[i:i+30] for i in range(0, len(attendance_text), 30)]
                for i, line in enumerate(attendance_lines):
                    cv2.putText(frame, line, (left + 6, bottom - 6 + i * text_height), font, 0.8, (255, 255, 255), 1)
            else:
                cv2.putText(frame, attendance_text, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    def log_attendance(self, user_id, name):
        current_schedule = self.db_manager.get_current_schedule(user_id)
        if current_schedule:
            schedule_id, start_time, end_time, course_name = current_schedule
            status = self.db_manager.log_attendance(user_id, schedule_id)
            if status == "Too soon to log again":
                self.result_label.config(text=f"Welcome, {name}! Attendance recently logged for {course_name}", fg="blue")
            else:
                self.result_label.config(text=f"Welcome, {name}! Attendance logged for {course_name}. Status: {status}", fg="green")
        else:
            
            user_name = self.db_manager.get_user_name(user_id)
            self.result_label.config(text=f"Welcome, {user_name}! No class scheduled.", fg="orange")

    def on_closing(self):
        if self.cap:
            self.cap.release()
        if self.db_manager:
            self.db_manager.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceScannerApp(root)
    root.mainloop()
