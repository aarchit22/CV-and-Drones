import cv2
import numpy as np

def colour(s):
    # Load the image
    image = cv2.imread(s)
    image=cv2.resize(image,(512,512))
    img_copy=image.copy()

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create a mask for the red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply the mask to the image
    red_image = cv2.bitwise_and(image, image, mask=mask)

    # Finding contours
    gray=red_image[:,:,2]
    edge=cv2.Canny(gray,60,450)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing contours
    cv2.drawContours(image,contours,-1,(0,255,0),2)

    # Display the images
    cv2.imshow('Original image',img_copy)
    cv2.imshow('Contoured Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

colour("perfect_shape.png")