import mediapipe as mp
from mediapipe.tasks.python import vision
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
