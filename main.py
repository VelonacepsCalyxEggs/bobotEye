import cv2 as cv
import numpy as np
import transmit
import time

import numpy as np
import cv2 as cv

def commandMode():
    print('Command mode active.')
    while True:
        transmit(input())


def detect_yellow_line(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the yellow line
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour which will be the yellow line
        largest_contour = max(contours, key=cv.contourArea)
        # Calculate the centroid of the contour
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Draw the centroid on the frame
            cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            return cx, cy, frame
    return None, None, frame

def follow_yellow_line(frame, transmit):
    MOVE_LEFT = 'D_200__'
    MOVE_FORWARD = 'W_100_R'
    MOVE_RIGHT = 'D_200_R'

    # Detect the yellow line and its centroid
    cx, cy, processed_frame = detect_yellow_line(frame)

    # Get the width of the frame
    _, width = frame.shape[:2]

    # Define the region in the middle of the frame where we want to keep the line
    middle_x_start = width // 3
    middle_x_end = 2 * width // 3

    # Check if we have detected the yellow line
    if cx is not None:
        # If the centroid is to the left of the middle region, move left
        if cx < middle_x_start:
            transmit.send(MOVE_LEFT)
        # If the centroid is to the right of the middle region, move right
        elif cx > middle_x_end:
            transmit.send(MOVE_RIGHT)
        # If the centroid is in the middle region, move forward
        else:
            transmit.send(MOVE_FORWARD)
    else:
        print("Yellow line not detected.")

    return processed_frame


def detect_yellow(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    height, width = mask.shape
    section_area = (width // 3) * (height // 3)  # Area of each section
    yellow_threshold = section_area * 0.2  # 20% of the section area

    # Initialize a list to hold the detection results
    detected_sections = []

    # Loop through each section
    for i in range(3):
        for j in range(3):
            # Calculate the coordinates of the current section
            x_start = j * width // 3
            x_end = (j + 1) * width // 3
            y_start = i * height // 3
            y_end = (i + 1) * height // 3

            # Extract the current section mask
            section_mask = mask[y_start:y_end, x_start:x_end]

            # Count the number of yellow pixels in the current section
            yellow_pixels = np.sum(section_mask == 255)

            # Check if yellow pixels exceed the threshold
            detected = (yellow_pixels > yellow_threshold)
            detected_sections.append(detected)

            # Draw a rectangle around the current section
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    return detected_sections, frame



x, y, w, h = 220, 140, 200, 200
capture = cv.VideoCapture(1)
if (input('Enter Y if you want to enter command mode: ')) == 'Y':
    commandMode()
while True:
    isTrue, frame = capture.read()
    #frame = frame[y:y+h, x:x+w] 
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([230, 255, 100])
    lower_white = np.array([0, 0, 50])
    upper_white = np.array([255, 255, 255])

    black_mask = cv.inRange(hsv, lower_black, upper_black)
    white_mask = cv.inRange(hsv, lower_white, upper_white)
    combined_mask = cv.bitwise_or(black_mask, white_mask)

    yellow = np.array([0, 255, 255], dtype=np.uint8)
    black = np.array([0, 0, 0], dtype=np.uint8)

    frame[white_mask == 255] = black
    frame[black_mask == 255] = yellow

    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    # Call the detect_yellow function and unpack the returned tuple
    #detected_sections, processed_frame = detect_yellow(frame)

    processed_frame = follow_yellow_line(frame, transmit)  # 'transmit' should be your communication object
    # if left:
    #     print(f"Yellow detected in left part: {left}")
    #     transmit.send('D_100__')
    # if middle:
    #     print(f"Yellow detected in middle part: {middle}")
    #     transmit.send('W_100_R')
    # if right:
    #     print(f"Yellow detected in right part: {right}")
    #     transmit.send('D_100_R')
    cv.imshow('framus', processed_frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
cv.waitKey(0)