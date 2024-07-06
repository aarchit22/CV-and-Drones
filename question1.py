import cv2
import numpy as np
import matplotlib.pyplot as plt

def hybrid(s1,s2):
    ft_filtered1=lpf(s1)
    ft_filtered2=hpf(s2)
    ft_hybrid=ft_filtered1+ft_filtered2
    hybrid=np.abs(np.fft.ifft2(np.fft.ifftshift(ft_hybrid)))
    plt.imshow(hybrid,cmap='gray')
    plt.show()

def lpf(s):
    # load the image
    image = cv2.imread(s, 0)
    image = cv2.resize(image, (256, 256))

    # performing fourier transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # creating rectangular lpf
    cutoff_radius=30
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    lpf = np.zeros((rows, cols))
    lpf[center_row - cutoff_radius:center_row + cutoff_radius,center_col - cutoff_radius:center_col + cutoff_radius] = 1

    # applying lpf
    f_transform_filtered = f_transform_shifted * lpf

    # inverse fourier transform
    img_filtered=np.abs(np.fft.ifft2(np.fft.ifftshift(f_transform_filtered)))

    # displaying as subplots
    plt.subplot(221),plt.imshow(image,cmap='gray'),plt.title('original')
    plt.subplot(222),plt.imshow(img_filtered,cmap='gray'),plt.title('filtered')
    plt.subplot(223), plt.imshow(np.log(1 + np.abs(f_transform_shifted)), cmap='gray'), plt.title('Shifted Fourier Transform')
    plt.subplot(224), plt.imshow(np.log(1 + np.abs(f_transform_filtered)), cmap='gray'), plt.title('Filtered Fourier Transform')
    plt.show()
    return f_transform_filtered

def hpf(s):
    # load image
    image_path = s
    image = cv2.imread(image_path, 0)
    
    image = cv2.resize(image, (256, 256))

    # Perform Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Create Rectangular HPF
    cutoff_radius = 25
    image_shape = image.shape
    rows, cols = image_shape
    center_row, center_col = rows // 2, cols // 2
    hpf = np.ones((rows, cols))
    hpf[center_row - cutoff_radius:center_row + cutoff_radius,center_col - cutoff_radius:center_col + cutoff_radius] = 0

    # applying the hpf
    f_transform_filtered = f_transform_shifted * hpf

    # Inverse Fourier Transform
    img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f_transform_filtered)))

    # Display the original and filtered images as subplots
    plt.subplot(221), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(222), plt.imshow(img_filtered, cmap='gray'), plt.title('Filtered')
    plt.subplot(223), plt.imshow(np.log(1 + np.abs(f_transform_shifted)), cmap='gray'), plt.title('Shifted Fourier Transform')
    plt.subplot(224), plt.imshow(np.log(1 + np.abs(f_transform_filtered)), cmap='gray'), plt.title('Filtered Fourier Transform')
    plt.show()
    return f_transform_filtered

hybrid("blonde.jpg","modi.png")