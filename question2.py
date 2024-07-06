import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel(s):
    img=cv2.imread(s,0)
    padded_img=np.pad(img,1,mode='edge')
    img_float=np.float32(padded_img)

    # sobel for vertical edges
    sobel_x=cv2.Sobel(img_float,cv2.CV_64F,1,0,ksize=3)

    # sobel for horizontal edges
    sobel_y=cv2.Sobel(img_float,cv2.CV_64F,0,1,ksize=3)

    # magnitude of gradient
    edge_detector=cv2.magnitude(sobel_x,sobel_y)

    # adjust intensity
    normalized_edge_detector=cv2.normalize(edge_detector,None,0,255,cv2.NORM_MINMAX)

    threshold=35
    rows,cols=img.shape
    edges=np.zeros((rows+2,cols+2))
    edges[normalized_edge_detector>threshold]=255

    edges_along_x=np.zeros((rows+2,cols+2))
    edges_along_x[cv2.normalize(np.abs(sobel_x),None,0,255,cv2.NORM_MINMAX)>threshold]=255

    edges_along_y=np.zeros((rows+2,cols+2))
    edges_along_y[cv2.normalize(np.abs(sobel_y),None,0,255,cv2.NORM_MINMAX)>threshold]=255

    # cv2.imshow('edges',edges)
    # cv2.imshow('edges along x',edges_along_x)
    # cv2.imshow('edges along y',edges_along_y)
    # cv2.imshow('Original',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.subplot(221),plt.imshow(img,cmap='gray'),plt.title('original')
    plt.subplot(222),plt.imshow(edges,cmap='gray'),plt.title('edges')
    plt.subplot(223),plt.imshow(edges_along_x,cmap='gray'),plt.title('vertical edges')
    plt.subplot(224),plt.imshow(edges_along_y,cmap='gray'),plt.title('horizontla edges')
    plt.show()
sobel('mujeres.png')