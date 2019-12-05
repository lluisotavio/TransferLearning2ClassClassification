import cv2, time
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocessVGG6
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
Config

video = cv2.VideoCapture(0)
model = load_model(Config.PATH_MODELO)
label_nome = {0:'nao_arma', 1:'arma'}

while True:
    check, frame = video.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    key = cv2.waitKey(1)
    color = cv2.resize(color, dsize=(224,224))
    color = np.expand_dims(color, axis=0)
    color = preprocessVGG6(color)
    label_previsto = model.predict(color)
    y_previsto = label_previsto.argmax()
    if y_previsto == 1:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'ARMA!!', (100, 130), font, 1,
                    (255, 255, 255), 2)
    cv2.imshow('preview', frame)


video.release()
cv2.destroyAllWindows()