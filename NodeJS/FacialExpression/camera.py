import cv2
from model import FacialExpressionModel
import numpy as np 

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel('checkpoints/model_final.json', 'checkpoints/model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
PIC_SIZE = 48

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (PIC_SIZE, PIC_SIZE))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()

if __name__ == '__main__':
    camera = VideoCamera()

    while True:
        camera.get_frame()

        if cv2.waitKey(1) == 27:
            break
    
    cv2.destroyAllWindows()