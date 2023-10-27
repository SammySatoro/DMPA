import cv2
import numpy as np

def task1():
    vid_cap = cv2.VideoCapture(0)

    while True:
        ret, frame = vid_cap.read()
        if not ret:
            exit()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", frame_hsv)
        cv2.imwrite("resources/task1.png", frame_hsv)
        if cv2.waitKey(1) & 0xFF == 27:
            exit()

def task2():
    vid_cap = cv2.VideoCapture(0)
    while True:
        ret, frame = vid_cap.read()
        if not ret:
            exit()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(frame_hsv, (0, 50, 50), (5, 255, 150))
        mask2 = cv2.inRange(frame_hsv, (175, 50, 50), (180, 255, 150))
        mask = cv2.bitwise_or(mask1, mask2)
        croped = cv2.bitwise_and(frame, frame, mask=mask)  # applies the binary mask to the original frame image
        cv2.imshow("HSV", croped)
        if cv2.waitKey(1) & 0xFF == 27:
            exit()

def task3():
    vid_cap = cv2.VideoCapture(0)
    lower_red = np.array([0, 110, 0])
    upper_red = np.array([15, 200, 255])
    while True:
        ret, frame = vid_cap.read()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        range = cv2.inRange(frame_hsv, lower_red, upper_red)
        kernel = np.ones((5, 5), np.uint8)

        opening = cv2.morphologyEx(range, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(range, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("open", opening)
        cv2.imshow("close", closing)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid_cap.release()
    cv2.destroyAllWindows()

task3()