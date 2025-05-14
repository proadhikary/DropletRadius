import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the current working directory
currentFolderPath = os.getcwd()

# Load the video
cap = cv2.VideoCapture("Data/M1.37AnimationNew.avi")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = 600
frame_height = 400
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_time_interval = 1 / fps  # time between frames in seconds

# Output video setup
output_path = os.path.join(currentFolderPath, "M1.37Animation.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Data lists
frame_times = []
b_values = []
frame_index = 0

# Frame loop
while True:
    ret, image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    _, _, _, heightC = cv2.boundingRect(largest_contour)
    measure = heightC / 4

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            cv2.drawContours(gray, [contour], -1, (255, 255, 255), 8)

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest_contour)
    C = (0.0002 * height) / measure
    B = C / 2

    # Store frame time and B value
    current_time = frame_index * frame_time_interval
    frame_times.append(current_time)
    b_values.append(B)

    print(f"Time = {current_time:.2f}s, C = {C:.5f}, B = {B:.5f}")

    # Draw and annotate
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 10)
    cv2.putText(image, f"B = {B:.5f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    image = cv2.resize(image, (frame_width, frame_height))
    out.write(image)
    cv2.imshow("Contours under Contours", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Save data to CSV
df = pd.DataFrame({
    'Time (s)': frame_times,
    'B Value': b_values
})
csv_path = os.path.join(currentFolderPath, 'M1.37AnimationNew.csv')
df.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(frame_times, b_values, marker='o')
plt.title('B Value over Time')
plt.xlabel('Time (s)')
plt.ylabel('B Value')
plt.grid(True)
plt.tight_layout()
plt_path = os.path.join(currentFolderPath, 'M1.37AnimationNew.png')
plt.savefig(plt_path)
plt.show()
