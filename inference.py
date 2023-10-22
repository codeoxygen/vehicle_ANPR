import cv2

sample_video = 'sample_video.mp4'

cap = cv2.VideoCapture(sample_video)

fps = cap.get(cv2.CAP_PROP_FPS)
print('frames per second =',fps)

frame_nb =1420

frame_count = 0

while True:
    
    ret, frame = cap.read()

    if not ret:
        break  

    if frame_count == frame_nb :
        image = frame
        break

    frame_count+=1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.imwrite('sample_frame.jpg' , image)

