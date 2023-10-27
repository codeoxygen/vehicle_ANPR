from flask import Flask, render_template, request, jsonify
import cv2
import torch
import hashlib
import os
import threading
import math
import cv2
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
from OCR_model import get_licsence_number , pattern

from PIL import Image


app = Flask(__name__)

# Load your custom-trained YOLOv5 model with your weights
# Update the absolute path to your custom YOLOv5 model code directory
yolov5_code_directory = 'D:\\Vehicle detection Project\\yolov5'

# Load your custom-trained YOLOv5 model with your weights
weights_path = 'D:\\Vehicle detection Project\\yolov5\\runs\\train\\exp\\weights\\best_with_200_epochs.pt'
weights_path_speed = 'D:\\Vehicle detection Project\\best_speed_weights.pt'
model = torch.hub.load(yolov5_code_directory, 'custom', path=weights_path, source='local')
model_speed = torch.hub.load(yolov5_code_directory, 'custom', path=weights_path_speed, source='local')

# Define vehicle labels based on the provided class mapping
class_mapping_nor = {"bus": 0, "car": 1, "threewheel": 2, "van": 3, "motorbike": 4}
vehicle_labels_nor = [label for label, index in sorted(class_mapping_nor.items(), key=lambda x: x[1])]

class_mapping_speed = {"lorry": 0, "car": 1, "bus": 2, "van": 3, "truck": 4, "double_cab": 5}
vehicle_labels_speed = [label for label, index in sorted(class_mapping_speed.items(), key=lambda x: x[1])]

# Video input path
video_path = 'D:\\Vehicle detection Project\\yolov5\\runs\\detect\\exp4\\rain.mp4'
video_path_speed = 'D:\\Vehicle detection Project\\High Speed Rain.mp4'

# Define the length of the axes lines and axis label offset outside the video frame
axis_label_offset = 10

# Initialize variables for violation counting and tracking
violations = 0
vehicle_ids = set()
line_color = (0, 255, 0)  # Red color for the line

# Initialize variables to capture frames before, during, and after violations
before_violation_frames = []
during_violation_frames = []
after_violation_frames = []

during_folder = 'during_violation_frames'
os.makedirs(during_folder, exist_ok=True)

violation_in_progress = False
detection_thread = None
detection_thread_speed = None 

nb_model = YOLO("best.pt")

video_capture = cv2.VideoCapture('sample_video.mp4')
tracker = cv2.TrackerKCF_create()

ocr = PaddleOCR(use_angle_cls=True, lang='en') 
# Initialize a dictionary to store speed information for each vehicle
vehicle_speeds = {}
# Initialize previous vehicle locations for speed calculation
prev_vehicle_locations = {}
# Define the frame rate of your video
fps = 18
# Speed violation threshold in km/h
speed_threshold = 50

def estimate_speed(location1, location2, time_elapsed):
    x1, y1 = location1
    x2, y2 = location2
    d_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    ppm = 8.8
    d_meters = d_pixels / ppm
    speed_mps = d_meters / time_elapsed
    speed_kmph = speed_mps * 3.6
    return speed_kmph

def number_plate_detection(frame):

    
    results = nb_model.predict(frame, conf=0.05)
    thickness = 2
    color = (0, 255, 0)

    tensor = results[0].boxes.xyxy
    tensor = tensor.cpu()
    arr = tensor.numpy()
    x, y = arr.shape

    
    if int(x)>0:
        for i in range(x):

            x_min , y_min , x_max , y_max = arr[i]
            img = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
            cropped_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            #cv2.imwrite('temorary_cropped',cropped_image )
            cropped_image = cv2.resize(cropped_image, (224, 224),
                        interpolation = cv2.INTER_LINEAR)
            
            cv2.imwrite('cache_image.jpg', cropped_image)
            position = (50, 100)  
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            font_color = (255, 255, 0)  # BGR color
            line_type = 2
            
            plate_number = get_licsence_number(ocr , 'cache_image.jpg')

            if plate_number :               
                if len(plate_number.strip().split('-')) < 4:
                    number = pattern(plate_number)
                else:
                    number = plate_number
                cv2.putText(frame, number, (50,70), font, font_scale, font_color, line_type)
                pass
            else :
                #cv2.putText(frame, 'Too small to detect plates', position, font, 1, font_color, line_type)
                pass

        return number
    else:
        return None
            


def start_speed_detection():
    cap = cv2.VideoCapture(video_path_speed)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        frame_resized = cv2.resize(frame, (500, 500))
        
        # Calculate hash for the resized frame
        frame_hash = hashlib.sha1(frame_resized.tobytes()).hexdigest()

        # Inference using your custom-trained YOLOv5 model
        results = model_speed(frame_resized)  # Assuming your model accepts a frame as input

        # Get the detected frame with bounding boxes
        detected_frame = results.render()[0]

        
        for result_idx, result in enumerate(results.pred[0]):
            class_index = int(result[-1])
            if class_index >= 0 and class_index < len(vehicle_labels_speed):
                label = vehicle_labels_speed[class_index]
                box = result[:4]
                x1, y1, x2, y2 = map(int, box)

                time_elapsed = 1.0 / fps  # Assuming constant frame rate

                prev_location = prev_vehicle_locations.get(result_idx)

                if prev_location is not None:
                    speed = estimate_speed(prev_location, (x1, y1), time_elapsed)

                    vehicle_speeds[result_idx] = speed

                    if speed > speed_threshold:
                        # Save the frame as a violation image
                        # violation_image_filename = os.path.join(output_directory, f'violation_{uuid.uuid4()}.jpg')
                        # cv2.imwrite(violation_image_filename, frame_resized)
                        cv2.putText(frame_resized, f'{label} Speed: {speed:.2f} km/h (Violation)', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        cv2.putText(frame_resized, f'{label} Speed: {speed:.2f} km/h', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                prev_vehicle_locations[result_idx] = (x1, y1)

        if any(speed > speed_threshold for speed in vehicle_speeds.values()):
            # Display the frame only if there is at least one violation
            cv2.imshow("Speed Violations", frame_resized)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def start_detection():
    global violations, vehicle_ids, line_color, during_violation_frames, before_violation_frames, violation_in_progress
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        new_width = 1920  
        new_height = 1080  


        original_height, original_width, _ = frame.shape
        aspect_ratio = original_width / original_height

        # Calculate the new size while preserving the aspect ratio
        if new_width is not None:
            new_height = int(new_width / aspect_ratio)
        elif new_height is not None:
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        frame_resized = cv2.resize(frame, (new_width, new_height))

        if not ret:
            break
        
        #frame_resized = cv2.resize(frame, (500, 500))
        frame_hash = hashlib.sha1(frame_resized.tobytes()).hexdigest()
    
        results = model(frame_resized)
    
        detected_frame = results.render()[0]
        cv2.line(detected_frame, (1370, 270), (540, 900), line_color, 2)
        violation_detected = False
        for result in results.pred[0]:
            class_index = int(result[-1])
            if class_index >= 0 and class_index < len(vehicle_labels_nor):
                label = vehicle_labels_nor[class_index]
                confidence = result[4].item()
               
                box = result[:4]
                x1, y1, x2, y2 = map(int, box)
                if 300 <= y1 <= 450 and 270 <= x1 <= 280:
                    vehicle_id = result[-1].item()
                    if vehicle_id not in vehicle_ids:
                        violations += 1
                        vehicle_ids.add(vehicle_id)
                        violation_detected = True
                        line_color = (0, 0, 255)
    

        
        if violation_detected:
            frame_filename = os.path.join(during_folder, f'violation_{violations}.jpg')
            
            cv2.imwrite(frame_filename, detected_frame)              
    
        if violation_detected:
            if not violation_in_progress:
                during_violation_frames = []
            violation_in_progress = True
        else:
            if violation_in_progress:
                after_violation_frames = []
            violation_in_progress = False
    
        if violation_in_progress:
            during_violation_frames.append(frame_resized.copy())
        else:
            before_violation_frames.append(frame_resized.copy())
            
        if not violation_detected:
            line_color = (0, 255, 0)
    
        cv2.imshow('Vehicle Detection Results', detected_frame)  
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')  # Use the template name without the path


@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    global detection_thread
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = threading.Thread(target=start_detection)
        detection_thread.start()
        return jsonify({'message': 'Detection started.'})
    else:
        return jsonify({'message': 'Detection is already in progress.'})
    
@app.route('/start_detection_speed', methods=['POST'])
def start_detection_speed_route():
    global detection_thread_speed
    if detection_thread_speed is None or not detection_thread_speed.is_alive():
        detection_thread_speed = threading.Thread(target=start_speed_detection)
        detection_thread_speed.start()
        return jsonify({'message': 'Speed Detection started.'})
    else:
        return jsonify({'message': 'Speed Detection is already in progress.'})

    
if __name__ == '__main__':
    app.run(debug=True)
