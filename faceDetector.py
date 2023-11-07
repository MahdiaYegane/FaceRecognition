import cv2
from cvzone.FaceDetectionModule import FaceDetector
import mediapipe as mp

video=cv2.VideoCapture("E:/openCV_images/1.mp4")
video.set(3,640)
video.set(4,480)

detector=FaceDetector(minDetectionCon=0.75)

while True:
    success,img=video.read()
    if success:
        img,boxs=detector.findFaces(img,draw=True)
        cv2.imshow("video",img)
        
        
        if cv2.waitKey(25)& 0xFF==ord("q"):           
            break 
        
cv2.destroyAllWindows()
        

