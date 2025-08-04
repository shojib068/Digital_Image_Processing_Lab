import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
image_path = 'input_images/20701068_Q4_input.tif'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Sobel kernels

# Horizontal edge detection
sobel_x_kernel = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
], dtype=np.float32)

# Vertical Edge Detection
sobel_y_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Padding
padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
rows, cols = image.shape

sobel_x = np.zeros_like(image, dtype=np.float32)
sobel_y = np.zeros_like(image, dtype=np.float32)

# convolution
for i in range(rows):
    for j in range(cols):
        region = padded_image[i:i+3, j:j+3]
        gx = np.sum(sobel_x_kernel * region)
        gy = np.sum(sobel_y_kernel * region)
        sobel_x[i, j] = gx
        sobel_y[i, j] = gy


sobel_x_display = np.clip(np.abs(sobel_x), 0, 255).astype(np.uint8)
sobel_y_display = np.clip(np.abs(sobel_y), 0, 255).astype(np.uint8)

combined_xy = np.hstack((sobel_x_display, sobel_y_display))

# Display image
plt.figure(figsize=(10, 5))
plt.title("Sobel Horizontal Edge                             Sobel Vertical Edge ")
plt.imshow(combined_xy, cmap='gray')
plt.axis('off')
plt.show()
