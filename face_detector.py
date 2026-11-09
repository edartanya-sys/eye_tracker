import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Pos_Puplis():
    def __init__(self, left = [0.5, 0.5], right = [0.5, 0.5]):
        self.left = left
        self.right = right

    def set_position(self, left, right):
        self.left = left
        self.right = right

    def get_position(self):
        return (self.left, self.right)

position = Pos_Puplis()

model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode





def result_callback(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print("FaceLandmarker Result")
    if result.face_landmarks:

        landmarks = result.face_landmarks[0]
        left_pupil = landmarks[468]
        right_pupil = landmarks[473]

        if position.get_position() != (left_pupil, right_pupil):
            position.set_position([left_pupil.x, left_pupil.y], [right_pupil.x, right_pupil.y])
            print(f"x = {left_pupil}, y = {right_pupil}")





    else:
        print("Лицо не найдено")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces = 1,
    min_face_detection_confidence = 0.70,
    min_tracking_confidence = 0.70,
    min_face_presence_confidence = 0.70,
    result_callback=result_callback)

face = FaceLandmarker.create_from_options(options)