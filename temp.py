import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('C:/Users/91840/Desktop/subject1.jpg')
alpha=cv2.imread('C:/Users/91840/Desktop/alpha1.png')

# filtered=cv2.bitwise_and(img,img,mask=alpha)
# print(0)
# cv2.imshow('Filtered',filtered)
img_b,img_g,img_r=cv2.split(img)
mask_b, mask_g, mask_r = cv2.split(alpha)
filtered_b = cv2.bitwise_and(img_b, img_b, mask=mask_b)
filtered_g = cv2.bitwise_and(img_g, img_g, mask=mask_g)
filtered_r = cv2.bitwise_and(img_r, img_r, mask=mask_r)
filtered = cv2.merge([filtered_b, filtered_g, filtered_r])

cv2.imshow('Filtered',filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()