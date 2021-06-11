import mediapipe as mp
import time
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
        #print(results.multi_hand_landmarks)
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
                       #print(id, land_mark)
                 h,w,c = img.shape
                 center_x, center_y = int(land_mark.x*w), int(land_mark.y*h)
                 #print(id, center_x,center_y)
                 self.land_mark_list.append([id , center_x, center_y])
                 if draw:
                     cv2.circle(img, (center_x,center_y), 5, (255,55,55), cv2.FILLED)
        return self.land_mark_list             
    
def main():
    previous_time = 0
    current_time = 0
    vid = cv2.VideoCapture(0)
    detector = handDetector()
   
    while(True):
       success, img = vid.read()
       img = detector.Find_hands(img)
       land_mark_list = detector.Find_position(img)
      # print(land_mark_list[4])
       if len(land_mark_list)!=0 :
           print(land_mark_list[4])
       current_time= time.time()
       fps = 1/ (current_time - previous_time)
       previous_time = current_time
       
       
       cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (200,0,0),3)
       
       cv2.imshow("Image",img)
       if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    cv2.destroyAllWindows()
 


if __name__ == "__main__":
    main()
