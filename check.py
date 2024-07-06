import cv2
import numpy as np

# Read the image
img = cv2.imread('color.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold or edge detection to highlight the circles
# You can choose either thresholding or edge detection based on your image characteristics
# Example thresholding:
_, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Example edge detection using Canny:
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Choose one of the contours (you can modify this based on your criteria)
selected_contour = contours[0]

# Draw the contour with a thickness of 30
cv2.drawContours(img, [selected_contour], 0, (0, 255, 0), 7)

# Display the result
cv2.imshow('Contoured Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
