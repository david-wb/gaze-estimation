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

with torch.no_grad():
    eyenet = EyeNet(nstack=4, inp_dim=64, oup_dim=34).to(device)

    if os.path.exists('checkpoint'):
        checkpoint = torch.load('checkpoint')
        eyenet.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise Exception('Unable to find model checkpoint file. Please download it.')

    sample = dataset[0]
    x = torch.tensor([sample['img']]).float().to(device)
    heatmaps = sample['heatmaps']
    heatmaps_pred, landmarks_pred, gaze_pred = eyenet.forward(x)

    heatmaps_pred = heatmaps_pred.cpu().numpy()[0, -1, :]
    heatmaps_pred = [cv2.resize(x, (150, 90)) for x in heatmaps]
    landmarks_pred = landmarks_pred.cpu().numpy()[0, :]

    result = [gaussian_2d(w=75, h=45, cx=c[1], cy=c[0], sigma=3) for c in landmarks_pred]

    plt.figure(figsize=(8, 9))

    iris_center = sample['landmarks'][-2][::-1]
    iris_center *= 2
    img = sample['img']

    img_gaze_pred = img.copy()
    for (y, x) in landmarks_pred[0:32]:
        cv2.circle(img_gaze_pred, (int(x*2), int(y*2)), 2, (255, 0, 0), -1)
    draw_gaze(img_gaze_pred, iris_center, gaze_pred.cpu().numpy()[0, :], length=60, color=(255, 0, 0))

    img_gaze = img.copy()
    for (y, x) in sample['landmarks'][0:32]:
        cv2.circle(img_gaze, (int(x*2), int(y*2)), 2, (0, 255, 0), -1)
    draw_gaze(img_gaze, iris_center, sample['gaze'], length=60, color=(0, 255, 0))

    plt.subplot(321)
    plt.imshow(sample['full_img'])
    plt.title('Raw training image')

    plt.subplot(322)
    plt.imshow(img)
    plt.title('Preprocessed training image')

    plt.subplot(323)
    plt.imshow(np.mean(heatmaps[16:32], axis=0), cmap='gray')
    plt.title('Ground truth heatmaps')

    plt.subplot(324)
    plt.imshow(np.mean(result[16:32], axis=0), cmap='gray')
    plt.title('Predicted heatmaps')

    plt.subplot(325)
    plt.imshow(img_gaze)
    plt.title('Ground truth landmarks and gaze vector')

    plt.subplot(326)
    plt.imshow(img_gaze_pred)
    plt.title('Predicted landmarks and gaze vector')
    plt.show()