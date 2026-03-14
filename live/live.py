from ultralytics import YOLO
import cv2
import math
import os

cap = cv2.VideoCapture("BeforeSplit/DJI_20251028133726_0001_V.MP4")
model = YOLO("bestmodels/best14.pt")
classNames = ["Drone"]

cv2.namedWindow('Yolo Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Yolo Detection', 1280, 720)

while True:
    success, img = cap.read()
    results = model(img,conf=0.4)

    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                        
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            bb1 = x1,y1
            bb2 = x2,y2
            #coco_format_txt_export(bb1=bb1,bb2=bb2)
            #get_coordinates()

    cv2.imshow('Yolo Detection', img)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()