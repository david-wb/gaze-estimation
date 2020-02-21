from __future__ import print_function, division
import os
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
import json
from util.preprocess import preprocess_unityeyes_image


class UnityEyesDataset(Dataset):

    def __init__(self, img_dir: Optional[str] = None):

        if img_dir is None:
            img_dir = os.path.join(os.path.dirname(__file__), 'UnityEyes/imgs')

        self.img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.img_paths = sorted(self.img_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.json_paths = []
        for img_path in self.img_paths:
            idx = os.path.splitext(os.path.basename(img_path))[0]
            self.json_paths.append(os.path.join(img_dir, f'{idx}.json'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        full_img = cv2.imread(self.img_paths[idx])
        with open(self.json_paths[idx]) as f:
            json_data = json.load(f)

        eye_sample = preprocess_unityeyes_image(full_img, json_data)
        sample = {'full_img': full_img, 'json_data': json_data }
        sample.update(eye_sample)
        return sample