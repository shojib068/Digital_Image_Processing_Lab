import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworth_low_pass_filter(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    V, U = np.meshgrid(v, u)
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)
    
    # Butterworth filter
    H = 1 / (1 + (D / cutoff)**(2 * order))

    H_2ch = np.repeat(H[:, :, np.newaxis], 2, axis=2)
    return H_2ch

# Input image
img = cv2.imread('input_images/20701068_Q9_input.tif', 0)

# DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

cutoff_frequencies = [5, 15, 30, 80, 230]
order = 2
filtered_images = []

for cutoff in cutoff_frequencies:
    mask = butterworth_low_pass_filter(img.shape, cutoff, order)
    fshift = dft_shift * mask

    # Inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    filtered_images.append(img_back)

# Display images
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i, cutoff in enumerate(cutoff_frequencies):
    plt.subplot(2, 3, i+2)
    plt.imshow(filtered_images[i], cmap='gray')
    plt.title(f'Butterworth LPF (cutoff={cutoff})')
    plt.axis('off')

plt.tight_layout()
plt.show()
