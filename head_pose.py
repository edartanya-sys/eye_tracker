import cv2 as cv
import numpy as np


def get_head_pose(landmarks, frame_shape):

    height, width = frame_shape[:2]

    #первод тчек из нормализованного вида в пиксельный
    image_points = np.array([
        (landmarks[1][0] * width, landmarks[1][1] * height),
        (landmarks[152][0] * width, landmarks[152][1] * height),
        (landmarks[33][0] * width, landmarks[33][1] * height),
        (landmarks[263][0] * width, landmarks[263][1] * height),
        (landmarks[61][0] * width, landmarks[61][1] * height),
        (landmarks[291][0] * width, landmarks[291][1] * height)
    ], dtype=np.float64)

    #усредненная 3д миоделька
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)


    focal_length = width
    #праметры камеры
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion_coefficients = np.zeros((4, 1)) #коэф искажения камеры

    #расположение лица относительно камеры
    success, rotation_vector, translation_vector = cv.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    angles = cv.RQDecomp3x3(rotation_matrix)[0]

    pitch = angles[0] #Вверх, вниз x
    yaw = angles[1] #Влево,  вправо y
    roll = angles[2] #наклон к плечу z

    return pitch, yaw, roll
