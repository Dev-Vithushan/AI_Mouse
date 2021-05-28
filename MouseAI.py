import cv2
import numpy as np
import HandTracingModule as htm
import time
import autopy

#################################
wCam,hCam = 640,480
pTime = 0
detector = htm.handDetector(maxHands=1)
frameR = 100
################################

cap = cv2.VideoCapture(1)       #Video camera allocation

cap.set(3,wCam)                  #Setting the video camera to the width
cap.set(4,hCam)                  #setting the video camera to the height

wScr, hScr = autopy.screen.size()



while (True):

    #1. Find the hand Landmark
    success, img = cap.read()
    img =detector.findHands(img)
    lmList, bbox = detector.findPosition(img)


    #2. Get the tip of the index and middle fingers

    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        #3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wCam - frameR, hCam - frameR),
                      (255,0,255),2)
        #4. Only index fingers the moving mode
        if fingers[1]==1 and fingers[2] ==0:

            #5. Convert coordinates
            x3 = np.interp(x1,(frameR,wCam - frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam - frameR),(0,hScr))
    #6. Smotheen values

            #7. Move mouse
            autopy.mouse.move(wScr-x3,y3)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
    #8. Both index and middle fingers are up : CLicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8,12,img)
            print(length)

            if length <=40:
                cv2.circle(img, (lineInfo[4],lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

    #9. Find distance between fingers

    #10 click mouse the distance shorten

    #11. Frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)


    #12. Display
    cv2.imshow("Image",img)
    cv2.waitKey(100)





