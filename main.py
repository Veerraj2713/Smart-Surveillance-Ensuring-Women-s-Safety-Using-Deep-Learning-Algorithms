import cv2
import time
import mediapipe as mp
from mtcnn import MTCNN
from Person_Detection import detect_person
from gender_Detection import classify_gender
from emotion_Detection import EmotionDetector
from Centroid_Tracker import CentroidTracker
from SOS_Condition import is_female_surrounded
from pose import detect_action
from Telebot_alert import send_telegram_alert
from facial_expression import classify_face, draw_selected_landmarks
from video import process_video

import cv2
from glob import glob
from pathlib import Path

from model import Model
from utils import plot

import warnings
warnings.filterwarnings('ignore')

import os
import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Initialize video capture and tracker
path = r"vil01.mp4"
# path = r"D:/TelePics/gettyimages-1200156556-640_adpp.mp4"
# path = r"D:/TelePics/gettyimages-1200156335-640_adpp.mp4"
# path = 0
webcam = cv2.VideoCapture(path)
tracker = CentroidTracker()
detector = MTCNN()  # Initialize MTCNN for face detection
emotion_detector = EmotionDetector()  # Initialize the EmotionDetector

model = Model()

mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
if not webcam.isOpened():
    print("Could not open video")
    exit()

try:
    skip_frame = 2
    frame_count = -1

    while True:
        output_path = 'live_output.avi'
        cap = cv2.VideoCapture(path)  # Open the default camera
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
        print('FPS', fps)

        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        success, frame = cap.read()
        status, frame = webcam.read()


        while success and cap.isOpened():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = model.predict(frame)
            label = prediction['label']
            conf = prediction['confidence']
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.putText(frame, label.title(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if label == 'Violence':
                print("hell")
            cv2.imshow('Live Feed', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop recording
            #     break
            # out.write(frame)
            # success, frame = cap.read()
            # status, frame = webcam.read()



            # if not status:
            #     print("Failed to read frame from video")
            #     break

            # frame_count += 1
            # if frame_count % skip_frame != 0:
            #     continue

            

            # # Detect persons in the frame
            # person_boxes = detect_person(frame)
            # n = len(person_boxes)  # stores the number of persons

            # # Reset gender counts for the current frame
            # male_count = 0
            # female_count = 0
            # mbbox = []

            

            # # Update tracker with detected person bounding boxes
            # objects = tracker.update(person_boxes)
            # print(f"Number of detected persons: {len(person_boxes)}")
            # print(f"Number of tracked objects: {len(objects)}")

            

            # for i, (objectID, centroid) in enumerate(objects.items()):
            #     if objectID < len(person_boxes):
            #         x1, y1, x2, y2 = map(int, person_boxes[i])  # Ensure bounding box values are integers
            #         person_img = frame[y1:y2, x1:x2]
                    
            #         # Detect faces within the person bounding box using MTCNN
            #         faces = detector.detect_faces(person_img)

            #         if faces:
            #             face = faces[0]  # Take the first detected face
            #             x, y, width, height = face['box']
            #             face_img = person_img[y:y+height, x:x+width]

            #             # Detect and classify facial expression
            #             results = mp_holistic.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        
            #             if results.face_landmarks:
            #                 face_class = classify_face(results.face_landmarks)  # Use the imported classify_face function

            #                 # Perform gender classification
            #                 gender_label = classify_gender(face_img)
            #                 print(f"Gender for objectID {objectID}: {gender_label}")

            #                 # Perform emotion detection
            #                 emotion_label = emotion_detector.detect_emotions(face_img)
            #                 print(f"Emotion for objectID {objectID}: {emotion_label}")
                            

            #                 if gender_label:
            #                     # Update counts based on gender
            #                     if 'male' in gender_label:
            #                         male_count += 1
            #                         mbbox.append(person_img)
            #                     elif 'female' in gender_label:
            #                         female_bbox = person_img
            #                         female_count += 1

            #                     # Detect and classify pose
            #                     pose_action = detect_action(results.pose_landmarks)

                                

            #                     # Combine all detected features into the label
            #                     label = f'ID {objectID}: {gender_label}, {face_class}, {emotion_label}, {pose_action}'

            #                     # Draw facial landmarks
            #                     draw_selected_landmarks(face_img, results.face_landmarks)

            #                     # Alert condition: Female detected alone at night
            #                     if n == 1 and 'female' in gender_label and (time.localtime().tm_hour >= 18 or time.localtime().tm_hour < 6):
            #                         send_telegram_alert(frame, "Female detected alone at night!")
            #                         print("Alert sent: Female detected alone at night.")
            #                 else:
            #                     label = f'ID {objectID}: Unknown'
            #             else:
            #                 label = f'ID {objectID}: Person'

            #             # Annotate the frame
            #             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #             cv2.circle(frame, tuple(map(int, centroid)), 4, (255, 0, 0), -1)
            #             # cv2.putText(frame, viol_det, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
            #         else:
            #             print(f"Warning: No face detected for object ID {objectID}")
                        
            # This part detects and alerts if a female is surrounded by males.
            # if female_count == 1 and n > 2 and (face_class == 'Fear' or face_class == 'Distress'):
            #     if is_female_surrounded(female_bbox, mbbox):
            #         send_telegram_alert(frame, "Female surrounded by men, potential danger detected!")
            #         print("Alert sent: Female surrounded by men.")
            
            # # Display the counts of males, females, and persons for the current frame
            # count_text = f'Males: {male_count}  Females: {female_count}  Total Persons: {n}'
            # cv2.putText(frame, count_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the annotated frame
            cv2.imshow("Webcam/Video Feed", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()
