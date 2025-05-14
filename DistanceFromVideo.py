import os
import cv2
import numpy as np

# Get the current working directory (local folder path)
currentFolderPath = os.getcwd()

# Load the video
cap = cv2.VideoCapture("Data/input.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = 600
frame_height = 400
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"FPS: {fps}")

# Define the codec and create VideoWriter object
output_path = os.path.join(currentFolderPath, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'avc1' or 'H264'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop to read frames from the video
while True:
    ret, image = cap.read()  # Read a frame
    if not ret:  # Break the loop if no frame is returned (end of video)
        break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    _, _, _, heightC = cv2.boundingRect(largest_contour)
    measure = heightC / 4
    for contour in contours:
        # Get the bounding rectangle
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x, y, width, height = cv2.boundingRect(contour)
        
        # Check if the contour has 4 sides (rectangle)
        if len(approx) == 4:
            cv2.drawContours(gray, [contour], -1, (255, 255, 255), 8)  # Fill with white

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours and hierarchy again
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest_contour)
    C = (0.0002 * height) / measure
    B = C / 2

    print(f"C = {C:.5f}, B = {B:.5f}")

    # Draw rectangle around largest contour
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 10)

    # Annotate the B value on the image
    cv2.putText(image, f"B = {B:.5f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Resize image
    image = cv2.resize(image, (frame_width, frame_height))

    # Show the result
    cv2.imshow("Contours under Contours", image)

    # Write the frame to the output video
    out.write(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()