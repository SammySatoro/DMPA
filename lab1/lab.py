import math

import cv2
import numpy as np

def task2():
    img1 = cv2.imread("images/chebupizza.jpg", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("images/chebupizza.png", cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread("images/chebupizza.bmp", cv2.IMREAD_ANYDEPTH)

    for name, value in {"img1": img1, "img2": img2, "img3": img3}.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, value)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task3():
    vid_cap = cv2.VideoCapture("video/polish-cow.mp4", cv2.CAP_ANY)
    size = (480, 640)

    while True:
        ret, frame = vid_cap.read()
        if not ret:
            exit()

        frame = cv2.resize(frame, (size))
        frame2 = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        cv2.imshow("polish cow", frame2)
        if cv2.waitKey(1) & 0xFF == 27:
            exit()


def task4():
    vid_cap = cv2.VideoCapture("video/polish-cow.mp4", cv2.CAP_ANY)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("task4_dir/polish-cow2.mp4", fourcc, 25, (width, height))

    while True:
        ret, vid = vid_cap.read()
        if not ret:
            break
        cv2.imshow('recording', vid)
        video_writer.write(vid)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid_cap.release()  # is optional here
    video_writer.release()  # is optional here
    cv2.destroyAllWindows()


def task5():
    img1 = cv2.imread("images/chebupizza.jpg")
    img2 = cv2.cvtColor(cv2.imread("images/chebupizza.jpg"), cv2.COLOR_BGR2HSV)

    cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task6():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        image = np.ones((height, width, 3), dtype=np.uint8)

        center_x, center_y = width // 2, height // 2
        top_rect = [(center_x - 15, center_y - 15), (center_x + 15, center_y - 100)]
        bottom_rect = [(center_x - 15, center_y + 15), (center_x + 15, center_y + 100)]
        horizontal_rect = [(center_x - 80, center_y - 15), (center_x + 80, center_y + 15)]

        line_color = (0, 0, 255)
        line_thickness = 2

        for rect in [top_rect,bottom_rect,horizontal_rect]:
            cv2.rectangle(image, rect[0], rect[1], line_color, line_thickness)

        res_frame = cv2.addWeighted(frame, 1, image, 0.5, 0)

        cv2.imshow("cross", res_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def task7():
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("video/task7_video.mp4", fourcc, 25, (width, height))

    while True:
        ret, vid = cap.read()
        if not ret:
            break
        cv2.imshow('webcam video', vid)
        video_writer.write(vid)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  # is optional here
    video_writer.release()  # is optional here
    cv2.destroyAllWindows()

def task8():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        image = np.ones((height, width, 3), dtype=np.uint8)

        center_x, center_y = width // 2, height // 2
        top_rect = [(center_x - 15, center_y - 15), (center_x + 15, center_y - 100)]
        bottom_rect = [(center_x - 15, center_y + 15), (center_x + 15, center_y + 100)]
        horizontal_rect = [(center_x - 80, center_y - 15), (center_x + 80, center_y + 15)]

        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = image_hsv.shape
        central_pixel = image_hsv[center_y, center_x]

        hue = central_pixel[0]
        if (hue >= 0) and (hue < 30) or (150 <= hue <= 180):
            line_color = (0, 0, 255)
        elif (30 <= hue < 90):
            line_color = (0, 255, 0)
        else:
            line_color = (255, 0, 0)

        for rect in [top_rect, bottom_rect, horizontal_rect]:
            cv2.rectangle(image, rect[0], rect[1], line_color, 2)   # -1 to fill rectangles

        res_frame = cv2.addWeighted(frame, 1, image, 0.5, 0)

        cv2.imshow("cross", res_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()   # is optional here
    cv2.destroyAllWindows()

def task8_infernal():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        image = np.ones((height, width, 3), dtype=np.uint8)

        center_x, center_y = width // 2, height // 2
        radius = 75

        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = image_hsv.shape
        central_pixel = image_hsv[center_y, center_x]

        hue = central_pixel[0]
        if (hue >= 0) and (hue < 30) or (150 <= hue <= 180):
            color = (0, 0, 255)
        elif (30 <= hue < 90):
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        star_points = []
        for i in range(5):
            angle = 2 * math.pi * i / 5 - 1 / math.pi
            x = round(radius * math.cos(angle))
            y = round(radius * math.sin(angle))
            star_points.append((x + center_x, y + center_y))

        cv2.circle(image, [center_x, center_y], 75, color, 2)
        for idx, point in enumerate(star_points):
            cv2.line(image, point, star_points[idx - 2], color, 2)
            cv2.line(image, point, star_points[idx - 3], color, 2)

        res_frame = cv2.addWeighted(frame, 1, image, 0.5, 0)

        cv2.imshow("cross", res_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()   # is optional here
    cv2.destroyAllWindows()

def task9():
    cap = cv2.VideoCapture("http://192.168.1.67:8080/video")

    while True:
        rec, frame = cap.read()
        if not rec:
            break

        cv2.imshow("camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()   # is optional here
    cv2.destroyAllWindows()

task9()