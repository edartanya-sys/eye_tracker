import cv2 as cv
from face_detector import face, get_landmarks
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera")
    exit()


right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133,
             173, 157, 158, 159, 160, 161, 246]

left_eye = [362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398]

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    frame_timestamp_ms = time.monotonic_ns() // 1000000

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    face.detect_async(mp_image, frame_timestamp_ms)

    landmarks = get_landmarks()

    if landmarks is not None:
        height, width, _ = frame.shape


        for x, y in landmarks:
            px = int(x * width)
            py = int(y * height)
            cv.circle(frame, (px, py), 1, (180, 180, 180), -1)


        for eye in [right_eye, left_eye]:
            for index in eye:
                x, y = landmarks[index]
                px = int(x * width)
                py = int(y * height)
                cv.circle(frame, (px, py), 2, (0, 255, 0), -1)


        for index in range(468, 478):
            x, y = landmarks[index]
            px = int(x * width)
            py = int(y * height)
            cv.circle(frame, (px, py), 3, (0, 0, 255), -1)

    cv.imshow('Face Tracker', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face.close()
cv.destroyAllWindows()
