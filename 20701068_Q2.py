import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
image_path = 'input_images/cameraman.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 3*3 kernel
def gaussian_kernel(size=3, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

gaussian_k = gaussian_kernel(3, sigma=1.0)

# Convolution
gaussian_blurred_manual = cv2.filter2D(image, -1, gaussian_k)

# Display image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("After applying Gaussian Filter")
plt.imshow(gaussian_blurred_manual, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
