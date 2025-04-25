import mediapipe as mp 
import cv2 as cv

model1 = mp.solutions.hands
draw = mp.solutions.drawing_utils

#   model.FaceMesh(min_detection_confidence =0.5) as faceMesh:

video = cv.VideoCapture(0)
handModel = model1.Hands(min_detection_confidence =0.5)


while video.isOpened():
    ret,frame =video.read()
    if ret == True:
        image = cv.cvtColor(frame,cv.COLOR_RGB2BGR)

        resultHand = handModel.process(image)
        if resultHand.multi_hand_landmarks:
              for j in resultHand.multi_hand_landmarks:
                draw.draw_landmarks(image,j,model1.HAND_CONNECTIONS)

        image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
        cv.imshow("Wind",image)
        if cv.waitKey(25) & 0xff == ord("p"):
            break        
        
    else:
        break

video.release()
cv.destroyAllWindows()    
