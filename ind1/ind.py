import cv2
import numpy as np


def task1(file, tracker_type, bbox):
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL().create()  # norm-sbit
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF().create()  # norm-lost
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE.create()  # norm-lost
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT().create()  # norm-lost

    # Read video
    video = cv2.VideoCapture(r"videos\\" + file)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(r"output\video_" + file + tracker_type + ".mp4", fourcc, 90, (w, h))

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        return

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video")
        return


    # bbox = cv2.selectROI(frame, True)
    # print(bbox)

    tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)
        writer.write(frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    writer.release()


tracker_types = ['MIL', 'KCF', 'CSRT']
files = ['1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4']
bboxs = [(555, 620, 150, 170), (570, 260, 160, 150), (1268, 196, 127, 137), (957, 406, 124, 145), (1399, 330, 106, 108)]

for tracker_type in tracker_types:
    for file in files:
        bbox = bboxs[files.index(file)]
        task1(file, tracker_type, bbox)

