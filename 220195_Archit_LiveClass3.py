import cv2
cap=cv2.VideoCapture(0)
num=0
while True:
    ret,img=cap.read()
    k=cv2.waitKey(5)
    if k==27:break
    elif k==ord('s'):
        cv2.imwrite('color'+str(num)+'.png',img)
        print("image saved")
        num+=1

    cv2.imshow("image",img)
cap.release()
cv2.destroyAllWindows()