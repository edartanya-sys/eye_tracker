import cv2 as cv
from face_detector import face
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame ")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    frame_timestamp_ms = int(time.time() * 1000)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    rgb_frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
    face.detect_async(mp_image, frame_timestamp_ms)


    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
