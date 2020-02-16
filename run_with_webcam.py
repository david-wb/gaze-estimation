import torch
from torch.nn import DataParallel

from models.posenet import PoseNet
import os
import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils
torch.backends.cudnn.enabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

webcam = cv2.VideoCapture(0)

dirname = os.path.dirname(__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(dirname, 'lbpcascade_frontalface_improved.xml'))
landmarks_detector = dlib.shape_predictor(os.path.join(dirname, 'shape_predictor_5_face_landmarks.dat'))

posenet = PoseNet(nstack=4, inp_dim=64, oup_dim=18)

checkpoint = torch.load('checkpoint_g')
posenet.load_state_dict(checkpoint['model_state_dict'])

posenet = posenet.to(device)

def main():
    current_face = None
    landmarks = None
    eye_landmarks = None
    alpha = 0.97
    eye_smoothing = 0.5

    while True:
        _, frame_bgr = webcam.read()
        frame_bgr = imutils.resize(frame_bgr, width=800)
        orig_frame = cv2.UMat(frame_bgr.copy())
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.UMat(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        if len(faces):
            next_face = faces[0]

            (x, y, w, h) = next_face
            if current_face is not None:
                current_face = alpha * next_face + (1 - alpha) * current_face
            else:
                current_face = next_face

        if current_face is not None:
            #draw_cascade_face(current_face, frame)
            next_landmarks = detect_landmarks(current_face, gray)

            if landmarks is not None:
                landmarks = next_landmarks * alpha + (1 - alpha) * landmarks
            else:
                landmarks = next_landmarks

            #draw_landmarks(landmarks, frame)


        if landmarks is not None:
            eyes = segment_eyes(orig_frame.get(), landmarks)
            next_eye_landmarks = run_posenet(eyes)
            if eye_landmarks is not None:
                eye_landmarks = eye_smoothing * next_eye_landmarks + (1 - eye_smoothing) * eye_landmarks
            else:
                eye_landmarks = next_eye_landmarks
            for (x, y) in eye_landmarks:
                cv2.circle(orig_frame,
                           (int(round(x[0][0])), int(round(y[0][0]))), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        cv2.imshow("Webcam", orig_frame)
        cv2.waitKey(1)


def detect_landmarks(face, frame, scale_x=0, scale_y=0):
    """Detect 5-point facial landmarks for faces in frame."""
    (x, y, w, h) = (int(e) for e in face)
    rectangle = dlib.rectangle(x, y, x + w, y + h)
    face_landmarks = landmarks_detector(frame.get(), rectangle)
    return face_utils.shape_to_np(face_landmarks)


def draw_cascade_face(face, frame):
    (x, y, w, h) = (int(e) for e in face)
    # draw box over face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


def draw_landmarks(landmarks, frame):
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)


def segment_eyes(frame, landmarks, ow=150, oh=90):
    eyes = []

    # Segment eyes
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # Centre image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Rotate to be upright
        roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
        rotate_mat = np.asmatrix(np.eye(3))
        cos = np.cos(-roll)
        sin = np.sin(-roll)
        rotate_mat[0, 0] = cos
        rotate_mat[0, 1] = -sin
        rotate_mat[1, 0] = sin
        rotate_mat[1, 1] = cos
        inv_rotate_mat = rotate_mat.T

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        # Centre image
        centre_mat = np.asmatrix(np.eye(3))
        centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_centre_mat = np.asmatrix(np.eye(3))
        inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                             inv_centre_mat)
        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        if is_left:
            eye_image = np.fliplr(eye_image)
        eyes.append({
            'image': eye_image,
            'transform_inv': inv_transform_mat,
            'side': 'left' if is_left else 'right',
        })
    return eyes


def run_posenet(eyes, ow=150, oh=90):

    imgs = [eye['image'] for eye in eyes]

    with torch.no_grad():
        x = torch.tensor(imgs, dtype=torch.float32).to(device)
        heatmaps_pred, landmarks_pred = posenet.forward(x)
        landmarks = landmarks_pred.cpu().numpy()
        assert landmarks.shape == (2, 18, 2)
        result = []
        for i, landmarks in enumerate(landmarks):
            eye = eyes[i]

            for (y, x) in landmarks[8:17]:
                y = y*oh/45
                x = x*ow/75
                if eye['side'] == 'left':
                    x = ow - x
                (x, y, _) = np.matmul(eye['transform_inv'], np.array([[x], [y], [1.0]]))
                result.append((x[0][0], y[0][0]))
        return np.array(result)


if __name__ == '__main__':
    main()