import os

import torch
from datasets.unity_eyes import UnityEyesDataset
from torch.utils.data import DataLoader
from models.eyenet import EyeNet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import cv2
import argparse

# Set up pytorch
torch.backends.cudnn.enabled = False
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device', device)

# Set up cmdline args
parser = argparse.ArgumentParser(description='Trains an EyeNet model')
parser.add_argument('--nstack', type=int, default=3, help='Number of hourglass layers.')
parser.add_argument('--nfeatures', type=int, default=32, help='Number of feature maps to use.')
parser.add_argument('--nlandmarks', type=int, default=34, help='Number of landmarks to be predicted.')
parser.add_argument('--nepochs', type=int, default=10, help='Number of epochs to iterate over all training examples.')
parser.add_argument('--start_from', help='A model checkpoint file to begin training from. This overrides all other arguments.')
parser.add_argument('--out', default='checkpoint.pt', help='The output checkpoint filename')
args = parser.parse_args()


def validate(eyenet: EyeNet, val_loader: DataLoader) -> float:
    with torch.no_grad():
        val_losses = []
        for val_batch in val_loader:
            val_imgs = val_batch['img'].float().to(device)
            heatmaps = val_batch['heatmaps'].to(device)
            landmarks = val_batch['landmarks'].to(device)
            gaze = val_batch['gaze'].float().to(device)
            heatmaps_pred, landmarks_pred, gaze_pred = eyenet.forward(val_imgs)
            heatmaps_loss, landmarks_loss, gaze_loss = eyenet.calc_loss(
                heatmaps_pred, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze)
            loss = 1000 * heatmaps_loss + landmarks_loss + gaze_loss
            val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        return val_loss


def train_epoch(epoch: int,
                eyenet: EyeNet,
                optimizer,
                train_loader : DataLoader,
                val_loader: DataLoader,
                best_val_loss: float,
                checkpoint_fn: str,
                writer: SummaryWriter):

    N = len(train_loader)
    for i_batch, sample_batched in enumerate(train_loader):
        i_batch += N * epoch
        imgs = sample_batched['img'].float().to(device)
        heatmaps_pred, landmarks_pred, gaze_pred = eyenet.forward(imgs)

        heatmaps = sample_batched['heatmaps'].to(device)
        landmarks = sample_batched['landmarks'].float().to(device)
        gaze = sample_batched['gaze'].float().to(device)

        heatmaps_loss, landmarks_loss, gaze_loss = eyenet.calc_loss(
            heatmaps_pred, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze)

        loss = 1000 * heatmaps_loss + landmarks_loss + gaze_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hm = np.mean(heatmaps[-1, 8:16].cpu().detach().numpy(), axis=0)
        hm_pred = np.mean(heatmaps_pred[-1, -1, 8:16].cpu().detach().numpy(), axis=0)
        norm_hm = cv2.normalize(hm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_hm_pred = cv2.normalize(hm_pred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if i_batch % 20 == 0:
            cv2.imwrite('true.jpg', norm_hm * 255)
            cv2.imwrite('pred.jpg', norm_hm_pred * 255)
            cv2.imwrite('eye.jpg', sample_batched['img'].numpy()[-1] * 255)

        writer.add_scalar("Training heatmaps loss", heatmaps_loss.item(), i_batch)
        writer.add_scalar("Training landmarks loss", landmarks_loss.item(), i_batch)
        writer.add_scalar("Training gaze loss", gaze_loss.item(), i_batch)
        writer.add_scalar("Training loss", loss.item(), i_batch)

        if i_batch > 0 and i_batch % 20 == 0:
            val_loss = validate(eyenet=eyenet, val_loader=val_loader)
            writer.add_scalar("validation loss", val_loss, i_batch)
            print('Epoch', epoch, 'Validation loss', val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'nstack': eyenet.nstack,
                    'nfeatures': eyenet.nfeatures,
                    'nlandmarks': eyenet.nlandmarks,
                    'best_val_loss': best_val_loss,
                    'model_state_dict': eyenet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_fn)

    return best_val_loss


def train(eyenet: EyeNet, optimizer, nepochs: int, best_val_loss: float, checkpoint_fn: str):
    timestr = datetime.now().strftime("%m%d%Y-%H%M%S")
    writer = SummaryWriter(f'runs/eyenet-{timestr}')
    dataset = UnityEyesDataset()
    N = len(dataset)
    VN = 160
    TN = N - VN
    train_set, val_set = torch.utils.data.random_split(dataset, (TN, VN))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

    for i in range(nepochs):
        best_val_loss = train_epoch(epoch=i,
                                    eyenet=eyenet,
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    best_val_loss=best_val_loss,
                                    checkpoint_fn=checkpoint_fn,
                                    writer=writer)


def main():
    learning_rate = 4 * 1e-4

    if args.start_from:
        start_from = torch.load(args.start_from, map_location=device)
        nstack = start_from['nstack']
        nfeatures = start_from['nfeatures']
        nlandmarks = start_from['nlandmarks']
        best_val_loss = start_from['best_val_loss']
        eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
        optimizer = torch.optim.Adam(eyenet.parameters(), lr=learning_rate)
        eyenet.load_state_dict(start_from['model_state_dict'])
        optimizer.load_state_dict(start_from['optimizer_state_dict'])
    elif os.path.exists(args.out):
        raise Exception(f'Out file {args.out} already exists.')
    else:
        nstack = args.nstack
        nfeatures = args.nfeatures
        nlandmarks = args.nlandmarks
        best_val_loss = float('inf')
        eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
        optimizer = torch.optim.Adam(eyenet.parameters(), lr=learning_rate)

    train(
        eyenet=eyenet,
        optimizer=optimizer,
        nepochs=args.nepochs,
        best_val_loss=best_val_loss,
        checkpoint_fn=args.out
    )


if __name__ == '__main__':
    main()