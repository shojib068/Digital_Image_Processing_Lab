import cv2
import numpy as np
import matplotlib.pyplot as plt
# input image
img = cv2.imread('input_images/20701068_Q20_input.jpeg')
if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
pixel_vals = img.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Values of K
K_values = [2, 3, 4, 5]

plt.figure(figsize=(15, 5))

for idx, K in enumerate(K_values):
    _, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((img.shape))

    # Display images
    plt.subplot(1, len(K_values), idx + 1)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title(f'K = {K}')
    plt.axis('off')

plt.tight_layout()
plt.show()
