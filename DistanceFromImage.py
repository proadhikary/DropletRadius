import os
import cv2
import numpy as np

# Get the current working directory (local folder path)
currentFolderPath = os.getcwd()


# Load the image
image = cv2.imread("data/M1P35/P1_000001.png")

# image = cv2.GaussianBlur(image, (5, 5), 0)

# cv2.imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))  # Width=1, Height=20

# # Detect vertical lines using morphological operations
# vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

# # Subtract detected lines from the original image
# binary = cv2.subtract(binary, vertical_lines)

# cv2.imshow("process Image", binary)
# Find contours and hierarchy
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

_, _, _, heightC = cv2.boundingRect(largest_contour)
measure=(heightC)/4
for contour in contours:
    # Get the bounding rectangle
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    x, y, width, height = cv2.boundingRect(contour)
    
    # Check if the contour has 4 sides (rectangle)
    if len(approx) == 4:
        print(f"Contour dimensions - Width: {width}, Height: {height}, X:{x}, Y:{y}")
        # Draw over the rectangle (optional: fill it with white or background color)
        cv2.drawContours(gray, [contour], -1, (255, 255, 255), 8)  # Fill with w

            
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)



# cv2.imshow(binary)
# Find contours and hierarchy
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Get the bounding rectangle
    x, y, width, height = cv2.boundingRect(contour)
        
    if width >100 and height >100:
            
            # cv2.waitKey(0)
        print(f"Contour dimensions - Width: {width}, Height: {height}, X:{x}, Y:{y}")
        C=(0.0002*height)/measure
        print(f"C= {C}")
            # Optionally, draw the rectangle on the image

        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 10)
            # cv2.imshow("Contours under Contours", image)
            



# # Draw contours based on hierarchy
# for i, contour in enumerate(contours):
#     if hierarchy[0][i][3] != -1:  # Check if the contour has a parent
#         cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw the child contour
image = cv2.resize(image, (600,400))
# Show the result
# cv2.imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()

