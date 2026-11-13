import mediapipe as mp
from threading import Lock

model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

landmarks_lock = Lock()
latest_landmarks = None


def result_callback(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks

    if result.face_landmarks:
        points = [(p.x, p.y) for p in result.face_landmarks[0]]

        with landmarks_lock:
            latest_landmarks = points
    else:
        with landmarks_lock:
            latest_landmarks = None


def get_landmarks():
    with landmarks_lock:
        if latest_landmarks is None:
            return None

        return latest_landmarks.copy()


def get_eye_features(landmarks):

    eyes = [
        {
            "corners": (33, 133),
            "top": (159, 160),
            "bottom": (145, 144),
            "iris": range(468, 473)
        },
        {
            "corners": (362, 263),
            "top": (386, 385),
            "bottom": (374, 380),
            "iris": range(473, 478)
        }
    ]

    features = []

    for eye in eyes:

        iris_x = sum(landmarks[i][0] for i in eye["iris"]) / 5
        iris_y = sum(landmarks[i][1] for i in eye["iris"]) / 5


        x1 = landmarks[eye["corners"][0]][0]
        x2 = landmarks[eye["corners"][1]][0]

        left_x = min(x1, x2)
        right_x = max(x1, x2)


        top_y = sum(landmarks[i][1] for i in eye["top"]) / 2
        bottom_y = sum(landmarks[i][1] for i in eye["bottom"]) / 2


        u = (iris_x - left_x) / (right_x - left_x)
        v = (iris_y - top_y) / (bottom_y - top_y)

        features.append((u, v))

    return features



options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.65,
    min_tracking_confidence=0.65,
    min_face_presence_confidence=0.65,
    result_callback=result_callback
)

face = FaceLandmarker.create_from_options(options)
