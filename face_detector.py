import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode



def result_callback(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            left_pupil = landmarks[468]
            print(f"Левый зрачок: x={left_pupil.x:.3f}, y={left_pupil.y:.3f}")
            right_pupil = landmarks[473]
            print(f"Правый зрачок: x={right_pupil.x:.3f}, y={right_pupil.y:.3f}")
        else:
            print("Лицо не найдено")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces = 1,
    min_face_detection_confidence = 0.65,
    min_tracking_confidence = 0.65,
    min_face_presence_confidence = 0.65,
    result_callback=result_callback)

face = FaceLandmarker.create_from_options(options)