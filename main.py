import cv2
import dlib
from imutils import face_utils, resize
from scipy.spatial import distance

class FaceDetector:
    def __init__(self, model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.l_start, self.l_end = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        self.r_start, self.r_end = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detector(gray, 0)
        return gray, subjects

    def get_eye_landmarks(self, shape):
        left_eye = shape[self.l_start:self.l_end]
        right_eye = shape[self.r_start:self.r_end]
        return left_eye, right_eye

class DrowsinessDetector:
    def __init__(self, threshold=0.25, frame_check=10):
        self.threshold = threshold
        self.frame_check = frame_check
        self.flag = 0

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_drowsiness(self, frame, left_eye, right_eye):
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.threshold:
            self.flag += 1
            if self.flag >= self.frame_check:
                return True
        else:
            self.flag = 0
        return False

if __name__ == '__main__':
    model_path = "./model/shape_predictor_68_face_landmarks.dat"
    face_detector = FaceDetector(model_path)
    drowsiness_detector = DrowsinessDetector()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = resize(frame, width=640, height=640)

        gray, subjects = face_detector.detect_faces(frame)
        for subject in subjects:
            shape = face_detector.predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)
            left_eye, right_eye = face_detector.get_eye_landmarks(shape)

            if drowsiness_detector.detect_drowsiness(frame, left_eye, right_eye):
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
