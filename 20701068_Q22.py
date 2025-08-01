import cv2
import numpy as np

# Input Image
img = cv2.imread('input_images/text.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binary threshold
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary_inv = cv2.bitwise_not(binary)

contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Regional descriptors
area = sum(cv2.contourArea(cnt) for cnt in contours)
perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
compactness = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0

# Euler number
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inv)
euler_number = n_labels - 1

# Texture
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_var = laplacian.var()

text_lines = [
    f"Area: {area:.2f} px",
    f"Perimeter: {perimeter:.2f} px",
    f"Compactness: {compactness:.4f}",
    f"Euler Number: {euler_number}",
    f"Texture: {laplacian_var:.2f}"
]
output_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
y_start, line_height = 50, 40
for i, line in enumerate(text_lines):
    y = y_start + i * line_height
    cv2.putText(output_img, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

# Save Output
cv2.imwrite("output_images/20701068_Q22_output.jpeg", output_img)
print("✅ Output saved as 'output_images/20701068_Q22_output.jpeg'")
