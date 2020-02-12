import torch
from datasets.unity_eyes import UnityEyesDataset
from torch.utils.data import DataLoader
from models.posenet import PoseNet
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

# default `log_dir` is "runs" - we'll be more specific here
timestr = datetime.now().strftime("%m-%d-Y-%H-%M-%S")
writer = SummaryWriter(f'runs/posenet')

dataset = UnityEyesDataset()
N = len(dataset)

VN = 100
TN = N - VN
train_set, val_set = torch.utils.data.random_split(dataset, (TN, VN))

dataloader = DataLoader(train_set, batch_size=4,
                        shuffle=True, num_workers=8)
valDataLoader = DataLoader(val_set, batch_size=4,
                        shuffle=True, num_workers=8)

posenet = PoseNet(nstack=8, inp_dim=256, oup_dim=32)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(posenet.parameters(), lr=learning_rate)


if os.path.exists('checkpoint'):
    checkpoint = torch.load('checkpoint')
    posenet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for i_batch, sample_batched in enumerate(dataloader):
    if i_batch % 20 == 0:
        torch.save({
            'model_state_dict': posenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint')

    print(i_batch, sample_batched['img'].size(),
          len(sample_batched['heatmaps']))

    X = torch.tensor(sample_batched['img'], dtype=torch.float32)
    Y = sample_batched['heatmaps']
    Yp = posenet.forward(X)

    loss = torch.mean(posenet.calc_loss(Yp, Y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i_batch, 'training loss', loss.item())
    writer.add_scalar("training loss", 1000 * loss.item(), i_batch)

    if i_batch % 10 == 0:
        with torch.no_grad():
            val_losses = []
            for val_batch in valDataLoader:
                X = torch.tensor(val_batch['img'], dtype=torch.float32)
                Y = val_batch['heatmaps']
                Yp = posenet.forward(X)
                loss = torch.mean(posenet.calc_loss(Yp, Y))
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            writer.add_scalar("validation loss", 1000 * val_loss, i_batch)
        print(i_batch, 'validation loss', val_loss)

