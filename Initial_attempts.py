import cv2 as cv
import numpy as np
import time
import threading
from queue import Queue

# Function to apply Gaussian blur and convert to HSV color space
def prepare_hsv_image(frame, kernel_size=(17, 17), sigmaX=0):
    blurred_frame = cv.GaussianBlur(frame, kernel_size, sigmaX)
    hsv = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)
    return hsv

def process_frames():
    while True:
        # Wait for a frame to be captured
        frame = frame_queue.get()
        # If 'None' frame received, it means the capture thread has finished
        if frame is None:
            break
        # Process the frame
        process_image(frame)
        # Display the result
        cv.imshow('Detection', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def capture_frames(interval, capture):
    while capture.isOpened():
        time.sleep(interval)
        ret, frame = capture.read()
        if not ret:
            break
        frame_queue.put(frame)

def process_image(frame):
    if frame is None:
        print("Error: No frame to process")
        return

    # Detect the square boundary
    square_coords = detect_square(frame)
    
    if square_coords:
        x, y, w, h = square_coords
        roi = frame[y:y+h, x:x+w]
        
        # Detect the cross within the square
        detect_cross(roi)
        
        # Detect the egg within the square
        detect_egg(roi)
        
        # Detect the white balls within the square
        detect_white_balls(roi)
        
        # Detect the orange ball within the square
        detect_orange_ball(roi)

def detect_square(frame):
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    hsv = prepare_hsv_image(frame)
    mask = cv.inRange(hsv, lower_red, upper_red)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            cv.drawContours(frame, [approx], 0, (0, 255, 0), 5)
            x, y, w, h = cv.boundingRect(approx)
            print(f"Square: {x}, {y}, {w}, {h}")
            return (x, y, w, h)
    return None

def detect_cross(frame):
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    hsv = prepare_hsv_image(frame)
    mask = cv.inRange(hsv, lower_red, upper_red)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
        if len(approx) == 12:
            cv.drawContours(frame, [approx], 0, (0, 255, 0), 5)
            x, y, w, h = cv.boundingRect(approx)
            print(f"Cross: {x}, {y}, {w}, {h}")
            break

def detect_white_balls(frame):
    hsv = prepare_hsv_image(frame)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    white_balls = []
    
    for contour in contours:
        area = cv.contourArea(contour)
        if 5 < area < 25:  # Area threshold to avoid noise and large objects like the egg
            (x, y), radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            white_balls.append((center, radius))
    
    for center, radius in white_balls:
        cv.circle(frame, center, radius, (255, 0, 0), 2)
    
    print(f"White Balls: {white_balls}")
    if not white_balls:
        print("No white balls detected.")

def detect_orange_ball(frame):
    hsv = prepare_hsv_image(frame)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    mask = cv.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    orange_ball = None
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 50:  # Area threshold to avoid noise
            (x, y), radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            orange_ball = (center, radius)
            break
    
    if orange_ball:
        center, radius = orange_ball
        cv.circle(frame, center, radius, (0, 255, 255), 2)
    else:
        print("No orange ball detected.")

def detect_egg(frame):
    hsv = prepare_hsv_image(frame)
    lower_egg = np.array([0, 0, 200])
    upper_egg = np.array([180, 55, 255])
    mask = cv.inRange(hsv, lower_egg, upper_egg)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    egg = None
    
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv.fitEllipse(contour)
            aspect_ratio = max(ellipse[1]) / min(ellipse[1])
            if 1.2 < aspect_ratio < 1.6 and cv.contourArea(contour) > 250:  # Additional check for contour area
                egg = ellipse
                break
    
    if egg is not None:
        cv.ellipse(frame, egg, (150, 50, 50), 2)
    else:
        print("No egg detected.")

# Create a queue for frame storage
frame_queue = Queue()

# Set up video capture
cap = cv.VideoCapture(1)

# Start the capture thread
capture_thread = threading.Thread(target=capture_frames, args=(2, cap))
capture_thread.start()

# Start the processing thread
processing_thread = threading.Thread(target=process_frames)
processing_thread.start()

# Wait for the threads to finish
capture_thread.join()
processing_thread.join()
cv.destroyAllWindows()

