import os
import cv2
import numpy as np

# Define the folder path containing the images
folder_path = "data/M1P35"

# List all PNG files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Loop through each image file
for filename in sorted(image_files):
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to create a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    largest_contour = max(contours, key=cv2.contourArea)
    _, _, _, heightC = cv2.boundingRect(largest_contour)
    measure = heightC / 4

    # Remove unwanted 4-sided contours from the gray image
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            cv2.drawContours(gray, [contour], -1, (255, 255, 255), 8)

    # Threshold again after cleanup
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # External contour detection
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > 100 and height > 100:
            C = (0.0002 * height) / measure
            print(f"{filename}, {C:.6f}")