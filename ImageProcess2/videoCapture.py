import cv2

#cap = cv2.VideoCapture(0)

highway = cv2.VideoCapture("Resources/highway.mp4")

count =0

# Object detection from stable camera

object_detector  = cv2.createBackgroundSubtractorMOG2(history=10,varThreshold=70)

while True:

   # success, img = cap.read()
    success,highwayvideo = highway.read()

    height,width,_ = highwayvideo.shape

    #print(height,width)
   # Extract REgion of  interest

    roi = highwayvideo[110: 550,100:650]


    # Object Detection
    mask = object_detector.apply(highwayvideo)
    _,mask = cv2.threshold(mask,80,255,cv2.THRESH_BINARY)


    contors,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contors:
        # Calculate Area and remove small elements

        area = cv2.contourArea(cnt)

        if area  > 100:
            #cv2.drawContours(highwayvideo, [cnt], -1, (0, 255, 0), 2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(highwayvideo,(x,y),(x+w,y+h),(0,255,0),3)








   # imggray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   # imgblur = cv2.GaussianBlur(imggray,(7,7),0)
    #cv2.imshow("Video",imggray)
   # cv2.imshow("Blur",imgblur)
    cv2.imshow("Highway",highwayvideo)
    cv2.imshow("Mask", mask)
    cv2.imshow("Roi", roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
