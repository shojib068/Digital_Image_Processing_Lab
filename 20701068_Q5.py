import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Input Image
image_path = 'input_images/20701068_Q5_input.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to add salt and pepper noise
def add_salt_pepper_noise(img, prob):
    noisy = np.copy(img)
    num_pixels = img.size
    num_salt = int(prob * num_pixels / 2)
    num_pepper = int(prob * num_pixels / 2)

    # salt noise
    for _ in range(num_salt):
        i = random.randint(0, img.shape[0] - 1)
        j = random.randint(0, img.shape[1] - 1)
        noisy[i, j] = 255

    # pepper noise
    for _ in range(num_pepper):
        i = random.randint(0, img.shape[0] - 1)
        j = random.randint(0, img.shape[1] - 1)
        noisy[i, j] = 0

    return noisy

# Add noise
noise_probability = 0.1
noisy_image = add_salt_pepper_noise(image, noise_probability)

# Median filter (3*3 kernel )
denoised_image = cv2.medianBlur(noisy_image, 3)

# Display image
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title("Salt & Pepper Noise Added")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("After Median Filtering")
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
