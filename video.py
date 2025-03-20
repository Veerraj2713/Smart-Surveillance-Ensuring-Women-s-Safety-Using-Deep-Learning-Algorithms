import cv2
from glob import glob
from pathlib import Path

from model import Model
from utils import plot

model = Model()

def process_video(path):
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

    while success and cap.isOpened():
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame)
        label = prediction['label']
        conf = prediction['confidence']
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, label.title(), 
                            (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 0, 255), 2)

        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop recording
            break
        out.write(frame)
        success, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()



# //


# import cv2
# from model import Model

# model = Model()

# def process_video(output_path: str, label_callback):
#     cap = cv2.VideoCapture(0)  # Open the default camera
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
#     size = (width, height)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
#     print('FPS', fps)
    
#     out = cv2.VideoWriter(output_path, fourcc, fps, size)
#     success, frame = cap.read()

#     while success and cap.isOpened():
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         prediction = model.predict(frame)
#         label = prediction['label']
#         conf = prediction['confidence']
        
#         # Send the label and confidence to the callback
#         label_callback(label, conf)

#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         frame = cv2.putText(frame, label.title(), 
#                             (10, 50), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, 
#                             (0, 0, 255), 2)

#         cv2.imshow('Live Feed', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop recording
#             break
#         out.write(frame)
#         success, frame = cap.read()

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Define the callback function to handle the label and confidence
# def label_callback(label, conf):
#     # Here, you could send the label to another module, store it, or print it
#     print(f"Label: {label}, Confidence: {conf}")

# save_video_path = 'live_output.avi'
# process_video(save_video_path, label_callback)

# model = Model()
# def process_video(frame):
#     image = cv2.imread(frame)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     label = model.predict(image)['label']
#     r = label
#     # print(r)
#     # print(type(r))
#     return r

# process_video("C:/Users/chitr/Downloads/street-fight-02_orig.jpg")

