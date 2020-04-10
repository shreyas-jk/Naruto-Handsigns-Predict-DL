import numpy as np
import operator
import cv2
import sys, os
from keras.models import load_model
from keras.models import model_from_json
import json
from PIL import Image
import pygame

pygame.init()
screen = pygame.display.set_mode((900,900),pygame.RESIZABLE)

CLIP_X1 = 160
CLIP_Y1 = 140
CLIP_X2 = 400
CLIP_Y2 = 360

with open('model_in_json.json','r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(model_json)
loaded_model.load_weights('model_weights.h5')

cap = cv2.VideoCapture(0)

while True:
    _, FrameImage = cap.read()
    FrameImage = cv2.flip(FrameImage, 1)
    cv2.imshow("", FrameImage)
    cv2.rectangle(FrameImage, (CLIP_X1, CLIP_Y1), (CLIP_X2, CLIP_Y2), (0,255,0) ,1)

    ROI = FrameImage[CLIP_Y1:CLIP_Y2, CLIP_X1:CLIP_X2]
    ROI = cv2.resize(ROI, (64, 64)) 
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    #ROI= cv2.add(ROI,np.array([40.0]))
    _, output = cv2.threshold(ROI, 100, 255, cv2.THRESH_BINARY) # adjust brightness
    
    SHOWROI = cv2.resize(ROI, (256, 256)) 
    _, output2 = cv2.threshold(SHOWROI, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", output2)
    
    result = loaded_model.predict(output.reshape(1, 64, 64, 1))
    predict =   { 'bird':    result[0][0],
                  'dragon':    result[0][1],    
                  'horse':    result[0][2],
                  'monkey':    result[0][3],
                  'dog':    result[0][3],
                  'ox':    result[0][5],
                  'serpent':    result[0][6],
                  'ram':    result[0][7],
                  }
    
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True)
    
    if(predict[0][1] == 1.0):
        predict_img  = pygame.image.load(os.getcwd() + '/dataset/images/' + predict[0][0] + '.png')
    else:
        predict_img  = pygame.image.load(os.getcwd() + '/dataset/images/nosign.png')
    predict_img = pygame.transform.scale(predict_img, (900, 900))
    screen.blit(predict_img, (0,0))
    pygame.display.flip()
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('q'): # esc key
        break
            
pygame.quit()
cap.release()
cv2.destroyAllWindows()
