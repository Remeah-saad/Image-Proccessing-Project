import cv2
import numpy as np
import matplotlib.pyplot as plt
# for arrow coloring
beige = (140, 180, 210)
brown = (45, 75, 120)
dark_orange = (0, 90, 180)

# I used three points to calculate the angle
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 180
    cos_angle = np.dot(v1, v2) / (magnitude1 * magnitude2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# I used image moments to find the center of each arrow
def find_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return np.array([cx, cy])

# I used sharp contour points to find the arrow tip
def find_arrow_tip(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    points = approx.reshape(-1, 2)
    if len(points) < 3:
        return None, approx
    center = find_centroid(contour)
    if center is None:
        return None, approx
    candidates = []

    # I checked every contour corner for a possible arrow tip
    for i in range(len(points)):
        previous_point = points[i - 1]
        current_point = points[i]
        next_point = points[(i + 1) % len(points)]
        angle = calculate_angle(previous_point, current_point, next_point)
        if angle < 85:
            distance = np.linalg.norm(current_point - center)
            candidates.append((distance, current_point))
    if len(candidates) == 0:
        return None, approx

# I selected the sharp point farthest from the center
    candidates.sort(key=lambda x: x[0], reverse=True)
    arrow_tip = candidates[0][1]
    return arrow_tip, approx

# I compared the tip with the center to determine the direction
def determine_direction(center, tip):
    dx = tip[0] - center[0]
    dy = tip[1] - center[1]
    if abs(dx) > abs(dy):
        if dx > 0:
            return "RIGHT"
        else:
            return "LEFT"
    else:
        if dy > 0:
            return "DOWN"
        else:
            return "UP"

# Main function for detecting multiple arrows
def detect_arrows(image):
    output = image.copy()

    # I converted the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # I used Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # I used Otsu thresholding to separate arrows from the background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # I used morphological closing to clean small gaps
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # I used contours to find all objects
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    directions = []

    # I used a loop to process every detected arrow
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        # Find the center of this arrow
        center = find_centroid(contour)
        if center is None:
            continue
        # Find the tip of this arrow
        tip, approx = find_arrow_tip(contour)
        if tip is None:
            continue
        # Determine this arrow direction
        direction = determine_direction(center, tip)
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        # Store the result
        directions.append((x, y, direction))
        # Draw contour
        cv2.drawContours(output, [approx], -1, beige, 2)
        cv2.rectangle(output, (x, y), (x + w, y + h), beige, 2)
        cv2.circle(output, tuple(center), 6, brown, -1)
        cv2.circle(output, tuple(tip), 8, brown, -1)
        cv2.line(output, tuple(center), tuple(tip), beige, 3, cv2.LINE_AA)
        cv2.putText(output, direction, (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, dark_orange, 2, cv2.LINE_AA)
    # Sort results from top to bottom, then left to right
    directions.sort(key=lambda item: (item[1], item[0]))
    detected_directions = [item[2] for item in directions]
    return output, binary, detected_directions

# Load the input image
image_path = "Example 1.webp"
image = cv2.imread(image_path)
if image is None: 
    print("Error: Image not found.") 
else: 
    result, binary, directions = detect_arrows(image) 
    print("Detected Directions:", directions) 
 
    # Display results 
    plt.figure(figsize=(12, 6)) 
    plt.subplot(121) 
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    plt.title("Original Image") 
    plt.axis("off") 
    plt.subplot(122) 
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Arrow Detection Result")
    plt.axis("off")
    plt.show()