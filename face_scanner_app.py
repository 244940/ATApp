import cv2
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
from database_manager import DatabaseManager
import face_recognition
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

class FaceScannerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Face Scanner")
        
        # Create a label to display the video stream or a placeholder
        self.video_label = Label(root)
        self.video_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        # Create a label to display scanning result
        self.result_label = Label(root, text="")
        self.result_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Initialize database manager
        try:
            self.db_manager = DatabaseManager()
            # Load known faces from database
            self.known_face_encodings, self.known_face_names, self.known_face_ids = self.db_manager.load_known_faces()
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to connect to the database: {str(e)}")
            self.db_manager = None
            self.known_face_encodings, self.known_face_names, self.known_face_ids = [], [], []

        # Load pre-trained mask detection model
        try:
            self.mask_model = load_model('mask_detector.h5')
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load mask detection model: {str(e)}")
            self.mask_model = None

        # Try to set up video capture
        try:
            self.cap = cv2.VideoCapture(0)  
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
        except Exception as e:
            messagebox.showwarning("Camera Error", f"Failed to initialize camera: {str(e)}\nRunning without live video.")
            self.cap = None

        # Start the video loop or show a placeholder
        self.update_frame()
        
        # Set up closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process frame (face detection, recognition, mask detection)
                self.process_frame(frame)
                
                # Convert the frame to PhotoImage
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
        else:
            # Show a placeholder image or message
            self.video_label.config(text="No camera feed available")
        
        # Call this function again after 10 milliseconds
        self.root.after(10, self.update_frame)
        

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
                    attendance_text = f"No scheduled class at this time"
            
            # Mask detection status
            #mask_text = "Mask status: Unknown"
            #if self.mask_model:
            #    face_image = frame[top:bottom, left:right]
            #    mask_label, mask_probability = self.predict_mask(face_image)
            #    #mask_text = f"Mask: {mask_label} ({mask_probability:.2f})"

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw a label with the name above the face
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 10), font, 1.0, (255, 255, 255), 1)  # Name label
            
            # Attendance text (with possible word wrapping for long text)
            (text_width, text_height), _ = cv2.getTextSize(attendance_text, font, 0.8, 1)
            if text_width > (right - left):
                attendance_lines = [attendance_text[i:i + 30] for i in range(0, len(attendance_text), 30)]
                for i, line in enumerate(attendance_lines):
                    cv2.putText(frame, line, (left + 6, bottom - 6 + i * text_height), font, 0.8, (255, 255, 255), 1)
            else:
                cv2.putText(frame, attendance_text, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            # Draw the mask status below the attendance status
            #cv2.putText(frame, (left + 6, bottom + 25), font, 0.8, (255, 255, 255), 1)  # Mask status label


    def predict_mask(self, image):
        # Preprocess the image
        image = cv2.resize(image, (128, 128))  # Adjust size if needed
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = self.mask_model.predict(image)
        mask_probability = prediction[0][0]

        # Determine the label based on probability
        mask_label = "with_mask" if mask_probability > 0.2 else "without_mask"
        
        return mask_label, mask_probability

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
            # No scheduled class, just display the user's name
            user_name = self.db_manager.get_user_name(user_id)
            self.result_label.config(text=f"Welcome, {user_name}! No scheduled class at this time.", fg="orange")
    
    def on_closing(self):
        if self.cap:
            self.cap.release()
        if self.db_manager:
            self.db_manager.close()
        self.root.destroy()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceScannerApp(root)
    root.mainloop()