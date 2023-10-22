import cv2

image_path = 'cropped_liscence_plates/count_1.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (64, 64), 
               interpolation = cv2.INTER_LINEAR)
print(image.shape)
window_name = 'image'
  
# Using cv2.imshow() method 
# Displaying the image 
cv2.imshow(window_name, image) 
  
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 