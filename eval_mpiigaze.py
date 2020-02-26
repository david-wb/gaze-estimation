import torch
from datasets.mpii_gaze import MPIIGaze
from models.eyenet import EyeNet
import os
import numpy as np
import cv2
from util.preprocess import gaussian_2d
from matplotlib import pyplot as plt
import util.gaze

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = MPIIGaze()
checkpoint = torch.load('checkpoint.pt', map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'])

with torch.no_grad():
    errors = []

    print('N', len(dataset))
    for i, sample in enumerate(dataset):
        print(i)
        x = torch.tensor([sample['img']]).float().to(device)

        heatmaps_pred, landmarks_pred, gaze_pred = eyenet.forward(x)

        gaze = sample['gaze'].reshape((1, 2))
        gaze_pred = np.asarray(gaze_pred.cpu().numpy())

        if sample['side'] == 'right':
            gaze_pred[0, 1] = -gaze_pred[0, 1]

        angular_error = util.gaze.angular_error(gaze, gaze_pred)
        errors.append(angular_error)
        print('---')
        print('error', angular_error)
        print('mean error', np.mean(errors))
        print('side', sample['side'])
        print('gaze', gaze)
        print('gaze pred', gaze_pred)

        # landmarks_pred = np.asarray(landmarks_pred.cpu().numpy())[0, :]
        #
        # plt.figure(figsize=(8, 9))
        #
        # iris_center = landmarks_pred[-2][::-1]
        # iris_center *= 2
        # img = sample['img']
        #
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #
        # img_gaze_pred = img.copy()
        # util.gaze.draw_gaze(img_gaze_pred, iris_center, gaze_pred[0, :], length=60, color=(255, 0, 0))
        #
        # img_gaze = img.copy()
        # util.gaze.draw_gaze(img_gaze, iris_center, sample['gaze'], length=60, color=(0, 255, 0))
        #
        # plt.subplot(121)
        # plt.imshow(cv2.cvtColor(img_gaze, cv2.COLOR_BGR2RGB))
        # plt.title('True Gaze')
        #
        # plt.subplot(122)
        # plt.imshow(cv2.cvtColor(img_gaze_pred, cv2.COLOR_BGR2RGB))
        # plt.title('Predicted Gaze')
        #
        # plt.show()




