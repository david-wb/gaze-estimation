from time import sleep

import torch
from datasets.unity_eyes import UnityEyesDataset
from torch.utils.data import DataLoader
from models.posenet import PoseNet
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import cv2
from util.preprocess import gaussian_2d
from matplotlib import pyplot as plt
import dlib
import imutils
from imutils import face_utils

webcam = cv2.VideoCapture(0)

dirname = os.path.dirname(__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(dirname, 'lbpcascade_frontalface_improved.xml'))
landmarks_detector = dlib.shape_predictor(os.path.join(dirname, 'shape_predictor_5_face_landmarks.dat'))

def main():
    current_face = None
    landmarks = None
    alpha = 0.5

    while True:
        _, frame = webcam.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imutils.resize(frame, width=800)
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
            draw_cascade_face(current_face, frame)
            next_landmarks = detect_landmarks(current_face, gray, scale_x=frame.shape[1]/gray.shape[1], scale_y=frame.shape[0]/gray.shape[0])

            if landmarks is not None:
                landmarks = next_landmarks*alpha + (1 - alpha) * landmarks
            else:
                landmarks = next_landmarks

            draw_landmarks(landmarks, frame)

        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)


def detect_landmarks(face, frame, scale_x=0, scale_y=0):
    """Detect 5-point facial landmarks for faces in frame."""
    (x, y, w, h) = (int(e) for e in face)
    rectangle = dlib.rectangle(x, y, x + w, y + h)
    face_landmarks = landmarks_detector(frame, rectangle)
    return face_utils.shape_to_np(face_landmarks)



def draw_cascade_face(face, frame):
    (x, y, w, h) = (int(e) for e in face)
    # draw box over face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


def draw_landmarks(landmarks, frame):
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)


if __name__ == '__main__':
    main()