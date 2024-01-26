import numpy as np
import pandas as pd
import cv2
import torch
from matplotlib import pyplot as plt

model = torch.hub.load('ultralytics/yolov5','yolov5s',)

# Real Time Detection with YOLO
cap = cv2.VideoCapture(1)

cv2.namedWindow('Real Time Detection with YOLO', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real Time Detection with YOLO', 1920, 1080)  

while cap.isOpened():
    ret, frame = cap.read()
    
    results = model(frame)
    
    cv2.imshow('Real Time Detection with YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()