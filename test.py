import torch
from datasets.unity_eyes import UnityEyesDataset
from models.eyenet import EyeNet
import os
import numpy as np
import cv2
from util.preprocess import gaussian_2d
from matplotlib import pyplot as plt
from util.gaze import draw_gaze

device = torch.device('cpu')
dataset = UnityEyesDataset()
checkpoint = torch.load('checkpoint.pt', map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'])

with torch.no_grad():
    sample = dataset[2]
    x = torch.tensor([sample['img']]).float().to(device)
    heatmaps = sample['heatmaps']
    heatmaps_pred, landmarks_pred, gaze_pred = eyenet.forward(x)

    landmarks_pred = landmarks_pred.cpu().numpy()[0, :]

    result = [gaussian_2d(w=80, h=48, cx=c[1], cy=c[0], sigma=3) for c in landmarks_pred]

    plt.figure(figsize=(8, 9))

    iris_center = sample['landmarks'][-2][::-1]
    iris_center *= 2
    img = sample['img']

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_gaze_pred = img.copy()
    for (y, x) in landmarks_pred[-2:-1]:
        cv2.circle(img_gaze_pred, (int(x*2), int(y*2)), 2, (255, 0, 0), -1)
    draw_gaze(img_gaze_pred, iris_center, gaze_pred.cpu().numpy()[0, :], length=60, color=(255, 0, 0))

    img_gaze = img.copy()
    for (x, y) in sample['landmarks'][-2:-1]:
        cv2.circle(img_gaze, (int(x*2), int(y*2)), 2, (0, 255, 0), -1)
    draw_gaze(img_gaze, iris_center, sample['gaze'], length=60, color=(0, 255, 0))

    plt.subplot(321)
    plt.imshow(cv2.cvtColor(sample['full_img'], cv2.COLOR_BGR2RGB))
    plt.title('Raw training image')

    plt.subplot(322)
    plt.imshow(img, cmap='gray')
    plt.title('Preprocessed training image')

    plt.subplot(323)
    plt.imshow(np.mean(heatmaps[16:32], axis=0), cmap='gray')
    plt.title('Ground truth heatmaps')

    plt.subplot(324)
    plt.imshow(np.mean(result[16:32], axis=0), cmap='gray')
    plt.title('Predicted heatmaps')

    plt.subplot(325)
    plt.imshow(img_gaze, cmap='gray')
    plt.title('Ground truth landmarks and gaze vector')

    plt.subplot(326)
    plt.imshow(img_gaze_pred, cmap='gray')
    plt.title('Predicted landmarks and gaze vector')
    plt.show()