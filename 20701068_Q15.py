import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input Image
img = cv2.imread('input_images/20701068_Q15_input.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def plot_histogram(image, title):
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0,256])
        plt.plot(hist, color=color)
    plt.xlim([0,256])
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

# Histogram equalization
def hist_eq_rgb(image):
    r, g, b = cv2.split(image)
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    return cv2.merge((r_eq, g_eq, b_eq))

img_eq = hist_eq_rgb(img)

# Display Images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_eq)
plt.title('Histogram Equalized Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plot_histogram(img, 'Histogram Before Equalization')

plt.subplot(1, 3, 3)
plot_histogram(img_eq, 'Histogram After Equalization')

plt.tight_layout()
plt.show()
