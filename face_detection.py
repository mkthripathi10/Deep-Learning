import mediapipe as mp
import cv2 as cv

model = mp.solutions.face_detection
drawing_utils = mp.solutions.drawing_utils

with model.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face:
      video = cv.VideoCapture(0)
      try:
        while video.isOpened():
            re,frame = video.read()

            if re == True:
                flip = cv.flip(frame,1)

                #mediapipeis requred RGB
                image = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
                result = face.process(image)

                image = cv.cvtColor(image,cv.COLOR_RGB2BGR)

                if result.detections:
                    for i in result.detections:
                        drawing_utils.draw_detection(image,i)

                cv.imshow("opne",image)
                if cv.waitKey(25) & 0xff == ord("p"):
                    break
            
            else:
                break

        video.release()   
        cv.destroyAllWindows()
      except ValueError:
        print(ValueError)  


