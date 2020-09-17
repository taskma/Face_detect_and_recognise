

import cv2
import os


### SETTINGS
folder = "dataset/"
#How many face photos do you want to capture? For 100 picture almost takes 45 seconds
capturingPhotoSize = 100

# Fill User Names
class_names = ['User1', 'User2', 'User3', 'User4', 'User5']
# Fill , which users face will detect?
#For example User2 = "1" id
face_id = 1

folder += class_names[face_id]
filename = folder + "/User."
### SETTINGS


if not os.path.exists(folder):
    os.mkdir(folder)

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')



print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
cv2.waitKey(6000)

while(True):

    ret, img = cam.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(filename + str(face_id) + '.' + str(count) + ".jpg", img[y:y+h,x:x+w])

        cv2.imshow('image', img)
        print(count)

    k = cv2.waitKey(400) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= capturingPhotoSize:
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


