
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

def hough_liness(s):
    
    # Load the image
    img=imread(s,as_gray=False)
    image = imread(s, as_gray=True)

    # Apply edge detection (Canny)
    edges = canny(image, sigma=2, low_threshold=0.1, high_threshold=0.3)

    # Apply Standard Hough Transform
    h, theta, d = hough_line(edges)

    # Find the most prominent lines
    lines = hough_line_peaks(h, theta, d)
    plot_image(image, lines,img)


def plot_image(image, lines,img):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot the image with detected lines
        axes[1].imshow(img)
        for _, angle, dist in zip(*lines):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
            axes[1].plot((0, img.shape[1]), (y0, y1), 'r')
        axes[1].set_xlim((0, img.shape[1]))
        axes[1].set_ylim((img.shape[0], 0))
        axes[1].set_title('Image with Detected Lines')
        axes[1].axis('off')

        plt.show()

s= 'circles.jpg'
hough_liness(s)
