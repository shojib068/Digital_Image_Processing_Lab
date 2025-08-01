import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input Image
img = cv2.imread('input_images/cameraman.tif', cv2.IMREAD_GRAYSCALE)
L = 256

# Identity function
def identity_transform(img):
    return img.copy()

# Thresholding function
def thresholding_transform(img, r_thresh=128):
    return np.where(img <= r_thresh, 0, 255).astype(np.uint8)

# General contrast stretching
def general_contrast_stretch(img, r1, r2, s1, s2):
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)

    mask1 = img <= r1
    mask2 = (img > r1) & (img <= r2)
    mask3 = img > r2

    if r1 != 0:
        out[mask1] = (s1 / r1) * img[mask1]
    else:
        out[mask1] = s1

    if r2 != r1:
        out[mask2] = ((img[mask2] - r1) * (s2 - s1) / (r2 - r1)) + s1
    else:
        out[mask2] = s1

    if r2 != 255:
        out[mask3] = ((255 - s2) / (255 - r2)) * (img[mask3] - r2) + s2
    else:
        out[mask3] = s2

    return np.clip(out, 0, 255).astype(np.uint8)

identity_img = identity_transform(img)
threshold_img = thresholding_transform(img, r_thresh=128)
stretch_img = general_contrast_stretch(img, r1=50, r2=180, s1=0, s2=255)

# Display images
titles = ['Identity (r1=s1, r2=s2)', 
          'Thresholding (r1=r2, s1=0, s2=255)', 
          'General Contrast Stretching (r1<r2, s1<s2)']
images = [identity_img, threshold_img, stretch_img]

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
