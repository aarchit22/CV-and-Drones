import numpy as np
import matplotlib.pyplot as plt
import cv2

width, height = 600, 400

saffron = (255, 153, 51)
white = (255, 255, 255)
green = (19, 136, 8)

image = np.ones((height, width, 3), dtype=np.uint8)*255

saffron_height = height // 3
image[:saffron_height, :, :] = saffron

green_height = height // 3
image[2 * saffron_height:, :, :] = green

chakra_center = (width // 2, height // 2)
chakra_radius = min(width, height) // 6
num_spokes = 24

theta = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
spoke_length = chakra_radius

for angle in theta:
    x = int(chakra_center[0] + spoke_length * np.cos(angle))
    y = int(chakra_center[1] + spoke_length * np.sin(angle))
    cv2.line(image, chakra_center, (x, y), (0, 0, 128), thickness=1)

cv2.circle(image, chakra_center, chakra_radius, (0, 0, 128), thickness=1)

plt.imshow(image)
plt.axis('off')
plt.show()