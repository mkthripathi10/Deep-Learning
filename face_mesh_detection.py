import mediapipe as mp 
import cv2 as cv

model = mp.solutions.face_mesh
draw = mp.solutions.drawing_utils

#   model.FaceMesh(min_detection_confidence =0.5) as faceMesh:

video = cv.VideoCapture(0)
faceModel =  model.FaceMesh(min_detection_confidence =0.5) 


while video.isOpened():
    ret,frame =video.read()
    if ret == True:
        image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
        resultFace = faceModel.process(image)

        if resultFace.multi_face_landmarks:
            for i in resultFace.multi_face_landmarks:
                draw.draw_landmarks(image,i)

        image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
        cv.imshow("Wind",image)
        if cv.waitKey(25) & 0xff == ord("p"):
            break        
        
    else:
        break

video.release()
cv.destroyAllWindows()    
