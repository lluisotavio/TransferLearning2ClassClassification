import cv2, time
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocessVGG6
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import Config

video = cv2.VideoCapture(Config.PATH_VIDEO_PARA_TESTE)
model = load_model(Config.PATH_MODELO)
label_nome = {0:'nao_arma', 1:'arma'}

fourcc = cv2.VideoWriter_fourcc(*'XVID')

fps = video.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(Config.PATH_VIDEO_TESTE_SAVE_PATH, fourcc, fps, (int(video.get(3)), int(video.get(4))))

while (video.isOpened()):
    check, frame = video.read()
    if check == True:

        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        color = cv2.resize(color, dsize=(224,224))
        color = np.expand_dims(color, axis=0)
        color = preprocessVGG6(color)
        label_previsto = model.predict(color)
        y_previsto = label_previsto.argmax()
        if y_previsto == 1:
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame, 'ARMA!!', (100, 130), font, 5,
                        (0, 0, 255), 2)

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    cv2.imshow('preview', frame)


video.release()
out.release()
cv2.destroyAllWindows()