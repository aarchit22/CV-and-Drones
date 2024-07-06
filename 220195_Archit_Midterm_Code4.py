import cv2
import numpy as np

image=cv2.imread("final.png")
image=cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

_,thresh_image=cv2.threshold(img,220,255,cv2.THRESH_BINARY)
contours,heirarchy=cv2.findContours(thresh_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for i,contour in enumerate(contours):
    if i==0:
        continue
    epsilon=0.01*cv2.arcLength(contour,True)
    approx=cv2.approxPolyDP(contour,epsilon,True)

    cv2.drawContours(image,[approx],0,(150,150,150),3)

    x,y,w,h=cv2.boundingRect(approx)
    xmid=int(x+w/2)
    ymid=int(y+h/2)

    coords=(xmid,ymid)
    color=(0,0,0)
    font=cv2.FONT_HERSHEY_COMPLEX

    if(len(approx)==3):
        cv2.putText(image,"Triangle",coords,font,1,color,1)
    elif(len(approx)==4):
        cv2.putText(image,"Square",coords,font,1,color,1)
    elif(len(approx)==10):
        cv2.putText(image,"Star",coords,font,1,color,1)
    else:
        cv2.putText(image,"Circle",coords,font,1,color,1)

cv2.imshow("Original",image)
cv2.waitKey(0)
cv2.destroyAllWindows()