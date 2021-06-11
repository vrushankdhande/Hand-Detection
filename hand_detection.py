7# -*- coding: utf-8 -*-
"""
Created on Sun May  2 00:18:16 2021

@author: vrush
"""

import cv2
import mediapipe as mp
import time
mp_hands = mp.solutions.hands
previous_time = 0
current_time = 0

vid = cv2.VideoCapture(0)

hands = mp_hands.Hands()
mp_draw= mp.solutions.drawing_utils    
while(True):
       success, img = vid.read()
       img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       results = hands.process(img_RGB)
       print(results.multi_hand_landmarks)
       if results.multi_hand_landmarks:
           for hand_marks in results.multi_hand_landmarks:
               for id, land_mark in enumerate(hand_marks.landmark):
                   #print(id, land_mark)
                   h,w,c = img.shape
                   center_x, center_y = int(land_mark.x*w), int(land_mark.y*h)
                   print(id, center_x,center_y)
                   #if id ==0:
                   cv2.circle(img, (center_x,center_y), 4, (255,55,55), cv2.FILLED)
                       
               mp_draw.draw_landmarks(img,hand_marks, mp_hands.HAND_CONNECTIONS )
               
       current_time= time.time()
       fps = 1/ (current_time - previous_time)
       previous_time = current_time
       
       
       cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (200,0,0),3)
       
       cv2.imshow("Image",img)
       if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
cv2.destroyAllWindows()


