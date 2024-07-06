import cv2
import numpy as np

def shape(s):
    # Read the image
    img = cv2.imread(s)
    img=cv2.resize(img,(512,512))
    original_img = img.copy()  # Create a copy for marking shapes

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) and approximate the shapes
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    approx_shapes = [cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True) for cnt in filtered_contours]

    # Identify and print the names of shapes
    for i, shape in enumerate(approx_shapes):
        num_sides = len(shape)
        shape_name = "Unknown"
        if num_sides == 3:
            shape_name = "Triangle"
        elif num_sides == 4:
            shape_name = "Rectangle"
        elif num_sides == 5:
            shape_name = "Pentagon"
        elif num_sides == 6:
            shape_name = "Hexagon"
        elif num_sides == 7:
            shape_name = "Heptagon"
        elif num_sides == 8:
            shape_name = "Octagon"
        elif num_sides == 9:
            shape_name = "Nonagon"
        else:
            shape_name="Circle"
        # Get the bounding box for the shape
        x, y, w, h = cv2.boundingRect(shape)

        # Mark the shape on the original image
        cv2.drawContours(original_img, [shape], -1, (0, 255, 0), 2)

        # Add text label for the shape on the shape itself
        cv2.putText(original_img, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Sort contours by area in descending order
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    # Mark the centers of the largest two shapes
    for i in range(min(2, len(sorted_contours))):
        M = cv2.moments(sorted_contours[i])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Draw a circle at the center
        cv2.circle(original_img, (cx, cy), 5, (255, 255, 255), -1)

        # Mark the largest shapes
        cv2.putText(original_img, f"Largest-{i + 1}", (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # Display the marked image
    cv2.imshow("Original image",img)
    cv2.imshow('Identified Shapes', original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

shape("two.jpg")
