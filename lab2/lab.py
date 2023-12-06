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
            break

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(frame_hsv, (0, 50, 35), (10, 255, 150))
        mask2 = cv2.inRange(frame_hsv, (175, 50, 50), (180, 255, 150))
        mask = cv2.bitwise_or(mask1, mask2)
        cropped = cv2.bitwise_and(frame, frame, mask=mask)  # applies the binary mask to the original frame image
        cv2.imshow("cropped", cropped)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid_cap.release()
    cv2.destroyAllWindows()

def task3():
    vid_cap = cv2.VideoCapture(0)
    lower_red = np.array([0, 110, 0])
    upper_red = np.array([15, 200, 255])
    while True:
        ret, frame = vid_cap.read()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        range = cv2.inRange(frame_hsv, lower_red, upper_red)
        kernel = np.ones((5, 5), np.uint8)

        # Opening: It is used to remove noise and small objects from the foreground of the image.
        # It is achieved by performing an erosion followed by dilation.
        # The opening operation can be useful for tasks such as image denoising and edge detection.
        opening = cv2.morphologyEx(range, cv2.MORPH_OPEN, kernel)

        # Closing: It is used to close small holes in the foreground objects, or small black points on the object.
        # It is achieved by performing a dilation followed by erosion.
        # The closing operation can be useful for tasks such as image denoising and filling in gaps in objects.
        closing = cv2.morphologyEx(range, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("open", opening)
        cv2.imshow("close", closing)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid_cap.release()
    cv2.destroyAllWindows()

def task4():
    cap = cv2.VideoCapture(0)
    lower_red = np.array([0, 110, 0])
    upper_red = np.array([15, 200, 255])
    while True:
        ret, frame = cap.read()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        range = cv2.inRange(frame_hsv, lower_red, upper_red)
        cropped = cv2.bitwise_and(frame, frame, mask=range)
        cv2.imshow('cropped', cropped)

        moments = cv2.moments(range)
        area = moments['m00']
        print(area)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def task5():
    cap = cv2.VideoCapture(0)
    lower_red = np.array([0, 120, 100])
    upper_red = np.array([60, 255, 255])
    while True:
        ret, frame = cap.read()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        range = cv2.inRange(frame_hsv, lower_red, upper_red)
        moments = cv2.moments(range)
        area = moments['m00']
        print(area)

        if area > 0:
            cx = int(moments['m10'] / area)
            cy = int(moments['m01'] / area)
            width = height = int(np.sqrt(area))

            # cross
            # cv2.line(
            #     frame,
            #     (cx - (width // 16), cy - (height // 16)),
            #     (cx + (width // 16), cy + (height // 16)),
            #     (0, 0, 0),
            #     2
            # )
            # cv2.line(
            #     frame,
            #     (cx + (width // 16), cy - (height // 16)),
            #     (cx - (width // 16), cy + (height // 16)),
            #     (0, 0, 0),
            #     2
            # )


            cv2.line(
                frame,
                (cx - (width // 16), cy),
                (cx + (width // 16), cy),
                (0, 0, 0),
                2
            )
            cv2.line(
                frame,
                (cx, cy - (height // 16)),
                (cx, cy + (height // 16)),
                (0, 0, 0),
                2
            )
            #
            # cv2.rectangle(
            #     frame,
            #     (cx - (width // 16), cy - (height // 16)),
            #     (cx + (width // 16), cy + (height // 16)),
            #     (0, 0, 0),
            #     2
            # )

        cv2.imshow('rect', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def task5_red():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(frame_hsv, (0, 50, 50), (30, 255, 150))
        mask2 = cv2.inRange(frame_hsv, (175, 50, 50), (180, 255, 150))
        mask = cv2.bitwise_or(mask1, mask2)
        cropped = cv2.bitwise_and(frame, frame, mask=mask)
        moments = cv2.moments(mask)
        area = moments['m00']
        print(area)

        if area > 0:
            cx = int(moments['m10'] / area)
            cy = int(moments['m01'] / area)
            width = height = int(np.sqrt(area))

            cv2.rectangle(
                cropped,
                (cx - (width // 32), cy - (height // 32)),
                (cx + (width // 32), cy + (height // 32)),
                (0, 255, 0),
                2
            )

        cv2.imshow('rect', cropped)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


task5()