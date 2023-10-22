import cv2
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
from OCR_model import get_licsence_number , pattern

from PIL import Image

model = YOLO("best.pt")

video_capture = cv2.VideoCapture('sample_video.mp4')
tracker = cv2.TrackerKCF_create()

ocr = PaddleOCR(use_angle_cls=True, lang='en') 

detected_plates = []
while True:
    ret, frame = video_capture.read()

    thickness = 2
    color = (0, 255, 0)
    count = 0
    results = model.predict(frame, conf=0.05)

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
            print(f'Result : {plate_number}')

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
            

            if number not in detected_plates:
                    detected_plates.append(number)
            x_ , y_ ,w_ , h_ = int(x_min), int(y_min), int(x_max), int(y_max)
 
    cv2.imshow('Plate_detection' , frame)

    if not ret:
        df = pd.DataFrame(detected_plates )
        df.to_csv("detected_plates.csv" , index=False)
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        df = pd.DataFrame(detected_plates )
        df.to_csv("detected_plates.csv" , index=False)     
        break
df = pd.DataFrame(detected_plates )
df.to_csv("detected_plates.csv" , index=False)
video_capture.release()
cv2.destroyAllWindows()
