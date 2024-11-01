import cv2
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model

class FaceRecognitionUtil:
    def __init__(self):
        # Load pre-trained mask detector model
        #self.mask_detector = load_model("mask_detector.h5")
        self.mask_detector = load_model("mask_detector_mobilenetv2_v2.h5")


    def process_frame(self, frame, known_face_encodings, known_face_names):
        # Convert the frame from BGR to RGB (as face_recognition expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Extract the face region from the frame for mask detection
            face_image = frame[top:bottom, left:right]
            
            # Determine if the person is wearing a mask
            mask_label = self.detect_mask(face_image)

            if mask_label == "Mask":
                # If mask is detected
                # - Use a higher tolerance for recognition due to occlusion
                name = self.recognize_face(face_encoding, known_face_encodings, known_face_names, tolerance=0.9)
            else:
                # Perform regular face recognition
                name = self.recognize_face(face_encoding, known_face_encodings, known_face_names)

            names.append(name)

            # Draw bounding box and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({mask_label})", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return frame, names

    def recognize_face(self, face_encoding, known_face_encodings, known_face_names, tolerance=0.6):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        return name

    def detect_mask(self, face_image):
        # Resize the face region to 128x128 pixels as expected by the mask detection model
        face_image_resized = cv2.resize(face_image, (128, 128))
        face_image_resized = face_image_resized.astype("float32") / 255.0  
        face_image_resized = np.expand_dims(face_image_resized, axis=0)
        
        # Predict whether the face is wearing a mask
        (mask, without_mask) = self.mask_detector.predict(face_image_resized)[0]

        # Return the prediction: 'Mask' or 'No Mask'
        return "Mask" if mask > without_mask else "No Mask"


