import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
count = 40

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("saving image%d"%count)
        cv2.imwrite("data/static/static%d.jpg"%count, frame)     # save frame as JPEG file
        count +=1

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count==50:
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()