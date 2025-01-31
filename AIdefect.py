import cv2
import numpy as np

# Load the image
image_path = '1.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define intensity thresholds
dark_upper_gray = 70  # Upper bound for dark regions
bright_lower_gray = 100 # Lower bound for bright regions

# Create masks for dark and bright regions
dark_defect_mask = cv2.inRange(gray_image, 0, dark_upper_gray)  # Dark regions
bright_defect_mask = cv2.inRange(gray_image, bright_lower_gray, 255)  # Bright regions

# Find contours for dark and bright regions
dark_contours, _ = cv2.findContours(dark_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bright_contours, _ = cv2.findContours(bright_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define size constraints for bounding boxes (adjust as needed)
min_width, max_width = 1, 100  # Minimum and maximum width
min_height, max_height = 1, 100  # Minimum and maximum height

# Open a text file to save the bounding box cordinates
label_file_path = 'bounding_boxes.txt'
with open(label_file_path, 'w') as label_file:
    
    # Draw green bounding boxes for dark regions
    for contour in dark_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_width <= w <= max_width and min_height <= h <= max_height:
            # Save the bounding box coordinates for dark regions
            label_file.write(f"Dark: {x} {y} {w} {h}\n")
            # Draw green bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw red bounding boxes for bright regions
    for contour in bright_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_width <= w <= max_width and min_height <= h <= max_height:
            # Save the bounding box coordinates for bright regions
            label_file.write(f"Bright: {x} {y} {w} {h}\n")
            # Draw red bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the result
cv2.imshow("Defect Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = 'defect_detection_result.jpg'
cv2.imwrite(output_path, image)
