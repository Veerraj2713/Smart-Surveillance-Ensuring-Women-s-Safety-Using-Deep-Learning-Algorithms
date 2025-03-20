
from ultralytics import YOLO
import cv2
# Load YOLOv8 model for person detection
yolo_model = YOLO("yolo11n.pt")

def detect_person(frame):
    results = yolo_model(frame)
    person_boxes = []
    print(" detect_person")
    if len(results) > 0:
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    if box.cls == 0:  # 'person' class index
                        x1, y1, x2, y2 = box.xyxy[0]
                        person_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 52, 32), 2)
                        # Display the class name above the bounding box
                        label = "Person"  # Change if you want to dynamically use class labels
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 100), 2)

    return frame, person_boxes
