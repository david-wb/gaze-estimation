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

device = torch.device('cpu')
print(device)

# default `log_dir` is "runs" - we'll be more specific here
timestr = datetime.now().strftime("%m-%d-Y-%H-%M-%S")
writer = SummaryWriter(f'runs/posenet')

dataset = UnityEyesDataset()
N = len(dataset)

VN = 100
TN = N - VN
train_set, val_set = torch.utils.data.random_split(dataset, (TN, VN))


with torch.no_grad():
    posenet = PoseNet(nstack=8, inp_dim=128, oup_dim=18).to(device)

    if os.path.exists('checkpoint'):
        checkpoint = torch.load('checkpoint')
        posenet.load_state_dict(checkpoint['model_state_dict'])

    sample = train_set[0]
    x = torch.tensor([sample['img']]).float().to(device)
    y = sample['heatmaps']
    yp, landmarks_pred = posenet.forward(x)

    heatmaps = yp.cpu().numpy()[0, -1, :]
    heatmaps = [cv2.resize(x, (150, 90)) for x in heatmaps]
    landmarks = landmarks_pred.cpu().numpy()[0, :]

    #centers = [centroid(x) for x in heatmaps]
    result = [gaussian_2d(w=75, h=45, cx=c[1], cy=c[0], sigma=3) for c in landmarks]

    plt.figure(figsize=(12, 3))

    plt.subplot(141)
    plt.imshow(sample['full_img'])
    plt.subplot(142)
    plt.imshow(sample['img'])
    plt.subplot(143)
    plt.imshow(np.mean(y[8:16], axis=0), cmap='gray')
    plt.subplot(144)
    plt.imshow(np.mean(result[8:16], axis=0), cmap='gray')
    plt.show()