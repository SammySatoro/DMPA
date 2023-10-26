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
    size = (480, 320)

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
        cv2.imshow('polish cow', vid)
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
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        center_x, center_y = width // 2, height // 2
        vertical_left_top = [(center_x - 15, center_y - 100), (center_x - 15, center_y - 15)]
        vertical_right_top = [(center_x + 15, center_y - 100), (center_x + 15, center_y - 15)]
        vertical_top = [(center_x - 15, center_y - 100), (center_x + 15, center_y - 100)]
        vertical_left_bot = [(center_x - 15, center_y + 15), (center_x - 15, center_y + 100)]
        vertical_right_bot = [(center_x + 15, center_y + 15), (center_x + 15, center_y + 100)]
        vertical_bottom = [(center_x - 15, center_y + 100), (center_x + 15, center_y + 100)]
        horizontal_left = [(center_x - 80, center_y - 15), (center_x - 80, center_y + 15)]
        horizontal_right = [(center_x + 80, center_y - 15), (center_x + 80, center_y + 15)]
        horizontal_top = [(center_x - 80, center_y - 15), (center_x + 80, center_y - 15)]
        horizontal_bottom = [(center_x - 80, center_y + 15), (center_x + 80, center_y + 15)]

        lines = [vertical_left_top, vertical_right_top, vertical_left_bot, vertical_right_bot, vertical_top,
            vertical_bottom, horizontal_left, horizontal_right, horizontal_top, horizontal_bottom]

        line_color = (0, 0, 255)
        line_thickness = 2

        for line in lines:
            cv2.line(image, line[0], line[1], line_color, thickness=line_thickness)

        res_frame = cv2.addWeighted(frame, 1, image, 0.5, 0)

        cv2.imshow("cross", res_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


task6()
