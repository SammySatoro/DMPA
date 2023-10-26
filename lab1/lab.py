import cv2

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

    vid_cap.release()       # is optional here
    video_writer.release()  # is optional here
    cv2.destroyAllWindows()

def task5():
    pass

task4()
