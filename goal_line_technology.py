import numpy as np
import pandas as pd
import cv2
import torch
import math
from matplotlib import pyplot as plt

goal_line = 350

model = torch.hub.load('ultralytics/yolov5','custom', path='C:\\Kuliah\\Semester 5\\Visi Komputer\\Tugas Akhir\\YOLO\\yolov5\\runs\\train\\exp7\\weights\\last.pt', force_reload=True)

# # Testing YOLO with Videos
cap = cv2.VideoCapture("crossbarcam.mp4")

in_goal_area = False

goal_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    results = model(frame)

    cv2.line(frame,(0,goal_line),(1920,goal_line),(255,255,255),10)
    
    # print(results.pandas().xyxy[0].to_dict().values())
    try:
        x1, y1, x2, y2, _, _, _ = results.pandas().xyxy[0].to_dict().values()
        x1, y1, x2, y2 = x1[0], y1[0], x2[0], y2[0]
        #print(x1, y1, x2, y2)
        
        if y1 > goal_line:
            cv2.putText(frame, 'GOAL!', (460, 250), cv2.FONT_HERSHEY_SIMPLEX , 10, (255, 255, 255), 30, cv2.LINE_AA)
            
            if not in_goal_area:
                    goal_count += 1
                    in_goal_area = True
                    
        else:
            in_goal_area = False
                    
            current_goal_text = f'Goal Conceded: {goal_count}'
            cv2.putText(frame, current_goal_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    except:
        continue

    cv2.imshow('Goal Line Technology', np.squeeze(results.render()))
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

#print(results)

# Real Time Detection with YOLO
cap = cv2.VideoCapture(0)

cv2.namedWindow('Real Time Detection with YOLO', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real Time Detection with YOLO', 700, 400)  

while cap.isOpened():
    ret, frame = cap.read()
    
    results = model(frame)
    
    # print(results)
    
    cv2.line(frame,(0,332),(814,332),(255,255,255),10)
    cv2.line(frame,(177,350),(927,350),(255,255,255),1)
    cv2.imshow('Real Time Detection with YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
