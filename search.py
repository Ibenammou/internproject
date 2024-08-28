import os
import cv2
from keras.models import load_model
import numpy as np

# Disable GPU-related warnings (optional)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Path to the Haar Cascade file
cascade_path = r'C:\Users\Windownet\Downloads\haarcascade_frontalface_default.xml'
facedetect = cv2.CascadeClassifier(cascade_path)

# Initialize video capture with the correct camera index
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# Load the pre-trained Keras model
model = load_model('keras_Model.h5', compile=False)

# Directory structure for patient images
base_directory = r'C:\Users\Windownet\Downloads\chu_project'

# Function to load patient images from the database folder
def load_patient_images(base_directory):
    patient_images = []
    patient_info = []
    
    for department in os.listdir(base_directory):
        department_path = os.path.join(base_directory, department)
        if os.path.isdir(department_path):
            for service in os.listdir(department_path):
                service_path = os.path.join(department_path, service)
                if os.path.isdir(service_path):
                    for doctor in os.listdir(service_path):
                        doctor_path = os.path.join(service_path, doctor)
                        if os.path.isdir(doctor_path):
                            for patient in os.listdir(doctor_path):
                                patient_path = os.path.join(doctor_path, patient)
                                if os.path.isdir(patient_path):
                                    for file in os.listdir(patient_path):
                                        if file.endswith(('.png', '.jpg', '.jpeg')):
                                            img_path = os.path.join(patient_path, file)
                                            img = cv2.imread(img_path)
                                            img = cv2.resize(img, (224, 224))  # Resize to the model's input size
                                            patient_images.append(img)
                                            
                                            # Save patient info (name, service, etc.)
                                            patient_info.append({
                                                'name': patient,
                                                'service': service,
                                                'department': department,
                                                'floor': 'Floor info here',  # Modify this if floor info is available in folder names
                                                'image_path': img_path
                                            })
    return patient_images, patient_info

# Load all patient images and their info
patient_images, patient_info = load_patient_images(base_directory)

# Function to preprocess image for model prediction
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to the model's input size
    image = image.astype('float32') / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def get_className(classNo):
    if classNo == 1:
        return "maryam"
    elif classNo == 0:
        return "imane"
    elif classNo == 2:
        return "khalifa"

# Function to find the best match for the detected face
def find_best_match(face_img, patient_images):
    min_distance = float('inf')
    best_match_idx = None
    
    # Compare face image with each patient image
    for idx, patient_img in enumerate(patient_images):
        # Predict using the model
        prediction = model.predict(preprocess_image(face_img))
        patient_prediction = model.predict(preprocess_image(patient_img))
        
        # Calculate the Euclidean distance between the face and the patient image predictions
        distance = np.linalg.norm(prediction - patient_prediction)
        
        if distance < min_distance:
            min_distance = distance
            best_match_idx = idx
    
    return best_match_idx, min_distance

while True:
    success, imgOriginal = cap.read()
    faces = facedetect.detectMultiScale(imgOriginal, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = imgOriginal[y:y+h, x:x+w]
        best_match_idx, min_distance = find_best_match(crop_img, patient_images)
        
        if best_match_idx is not None:
            # Get the patient's information
            patient = patient_info[best_match_idx]
            print(f"Match found: {patient['name']}, Service: {patient['service']}, Department: {patient['department']}, Floor: {patient['floor']}")
            
            # Draw the bounding box and patient info on the frame
            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgOriginal, f"{patient['name']} ({patient['service']})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the video frame with predictions
    cv2.imshow("Result", imgOriginal)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
