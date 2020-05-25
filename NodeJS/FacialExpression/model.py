import numpy as np 
import seaborn as sb 
import keras
from keras.models import model_from_json


import matplotlib.pyplot as plt 

class FacialExpressionModel(object):
    EMOTION_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neurtral']

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)


        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        print('Model loaded from disk')
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)

        return FacialExpressionModel.EMOTION_LIST[np.argmax(self.preds)]