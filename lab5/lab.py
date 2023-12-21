import cv2

def task(kernel_size, standard_deviation, delta_thresh, min_area):

    video = cv2.VideoCapture('/home/sammysatoro/PycharmProjects/DMPA/lab5/videos/hand.mp4', cv2.CAP_ANY)

    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('/home/sammysatoro/PycharmProjects/DMPA/lab5/output/result.mp4', fourcc, 144, (w, h))

    while True:
        old_img = img.copy()
        ok, frame = video.read()
        if not ok:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

        diff = cv2.absdiff(img, old_img)

        thresh = cv2.threshold(diff, delta_thresh, 255, cv2.THRESH_BINARY)[1]

        (contors, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contr in contors:
            area = cv2.contourArea(contr)
            if area < min_area:
                continue
            video_writer.write(frame)

    video_writer.release()


kernel_size = 3
standard_deviation = 50
delta_thresh = 60
min_area = 20
task(kernel_size, standard_deviation, delta_thresh, min_area)

# kernel_size = 11
# standard_deviation = 70
# delta_thresh = 60
# min_area = 20
# task(kernel_size, standard_deviation, delta_thresh, min_area)
