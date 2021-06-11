import time
import mediapipe as mp
import os
import cv2

class handDetector():
    def __init__(self,mode=False,is_authenticated=True,maxHands=2,detection_confidence=0.5,tracking_confidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence= detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,self.maxHands,self.detection_confidence,self.tracking_confidence)
        self.mp_draw= mp.solutions.drawing_utils    
    
    def Find_hands(self, img, draw=True):
        img_RGB = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks:
            for hand_marks in self.results.multi_hand_landmarks:
               if draw:
                   self.mp_draw.draw_landmarks(img,hand_marks, self.mp_hands.HAND_CONNECTIONS )
        return img 
    
    def Find_position(self, img, hand_no = 0, draw= True):
        self.land_mark_list =[]
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, land_mark in enumerate(my_hand.landmark):
                 h,w,c = img.shape
                 center_x, center_y = int(land_mark.x*w), int(land_mark.y*h)
                 self.land_mark_list.append([id , center_x, center_y])
                 if draw:
                     cv2.circle(img, (center_x,center_y), 5, (255,55,55), cv2.FILLED)
        return self.land_mark_list      

wCam , hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folder_path = "nu"
my_list = os.listdir(folder_path)
print(my_list)

over_lay_list=[]
for img_path in my_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    print(f'{folder_path}/{img_path}')
    over_lay_list.append(image)
    print(len(over_lay_list))
    
print(len(over_lay_list))
previous_time = 0

detector = handDetector(detection_confidence=0.5)
Tip_IDS= [4, 8,12, 16,20]

while True:
   success, img = cap.read()
   img = detector.Find_hands(img)
   landmark_List = detector.Find_position(img,draw = False)
   
   if len(landmark_List)!=0 :
       Figures = []
       
       #thumb
       if landmark_List[Tip_IDS[0]][1]>landmark_List[Tip_IDS[0]-1][1]:
             Figures.append(1)
       else : 
           Figures.append(0)
    
    #for 4 fing
       for id in range(1,5):
           if landmark_List[Tip_IDS[id]][2]<landmark_List[Tip_IDS[id]-2][2]:
             Figures.append(1)
           else :
               Figures.append(0)
       total_figure= Figures.count(1)
       print(total_figure)
   
       h, w, c = over_lay_list[total_figure-0].shape
       img[0:h,0:w]=over_lay_list[total_figure-0]
       cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
       cv2.putText(img, str(total_figure), (45,375), cv2.FONT_HERSHEY_PLAIN,10,(255,89,45),25)
   
   current_time = time.time()
   fps = 1 / (current_time-previous_time)
   previous_time = current_time
   cv2.putText(img, f'FPS: {int(fps)}',(400, 55), cv2.FONT_HERSHEY_PLAIN , 3, (255,12,12),3)
   
   cv2.imshow("Image",img)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    
cv2.destroyAllWindows()