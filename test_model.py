from models.posenet import PoseNet
from util.preprocess import preprocess_unityeyes_image, get_heatmaps
from models.layers import Conv, Residual, Hourglass
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import json

posenet = PoseNet(nstack=8, inp_dim=256, oup_dim=32)

img1_path = './datasets/UnityEyes/imgs/1.jpg'
img1 = cv2.imread(img1_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
with open('./datasets/UnityEyes/imgs/1.json') as json_file:
    json_data = json.load(json_file)

eye_entry = preprocess_unityeyes_image(img1, json_data)
input_img = eye_entry['img'] / 255.

heatmaps_true = eye_entry['heatmaps']

inp = torch.tensor([input_img], dtype=torch.float32)
print(inp.shape)

preds = posenet.forward(inp)
trues = torch.Tensor([heatmaps_true])

print(preds.shape)
print(trues.shape)
loss = torch.mean(posenet.calc_loss(preds, trues))

loss.backward()
