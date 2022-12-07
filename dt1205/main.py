import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import os

en_dict = {}
path = 'C:/Users/User/Desktop/train_simplified/'

filenames = os.listdir(path)

print(filenames[:5])

def encode_labels():
    counter = 0
    for fn in filenames:
        en_dict[fn[:-4].split('/')[-1].replace(' ', '_')] = counter
        counter += 1

encode_labels()

dec_dict = {v: k for k , v in en_dict.items()}

def decode_labels(label):
    return dec_dict[label]

def get_label(nfile):
    #print(nfile[:-4].split('/')[-1].replace(' ', '_'))
    return en_dict[nfile[:-4].split('/')[-1].replace(' ', '_')]

import pandas as pd
import ast
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DoodleDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode='train', nrows=1000, skiprows=None, size=256, transform=None):
        self.root_dir = root_dir
        file = os.path.join(root_dir, csv_file)
        self.size = size
        self.mode = mode
        self.doodle = pd.read_csv(file, usecols=['drawing'], nrows=nrows, skiprows=skiprows)
        self.transform = transform
        if self.mode == 'train':
            self.label = get_label(csv_file)

    @staticmethod
    def _draw(raw_strokes, size=256, lw=6, time_color=True):
        BASE_SIZE = 256
        img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)

        for t, stroke in enumerate(raw_strokes):
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 if time_color else 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

        if size != BASE_SIZE:
            return cv2.resize(img, (size, size))

        else:
            return img

    def __len__(self):
        return len(self.doodle)

    def __getitem__(self, index):
        raw_strokes = ast.literal_eval(self.doodle.drawing[index])
        sample = self._draw(raw_strokes, size=self.size, lw=2, time_color=True)

        if self.transform:
            sample = self.transform(sample)

        if self.mode == 'train':
            return (sample[None] / 255).astype('float32'), self.label
        else:
            return (sample[None] / 255).astype('float32')

SIZE = 224
select_nrows= 10000

doodles = ConcatDataset([DoodleDataset(fn.split('/')[-1], path, mode='train', nrows=select_nrows, skiprows=None, size=SIZE, transform=None) for fn in filenames])

train_dataloader = DataLoader(doodles, batch_size=32, shuffle=True, num_workers=0)
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

for images, label in train_dataloader:
    break

plt.figure(figsize=(16,24))
imshow(torchvision.utils.make_grid(images[:24]))


def validation(get_loader, lossfn, scorefn):
    model.eval()
    loss, score = 0, 0
    vlen = len(get_loader)

    for X, y in get_loader:
        X = X.to(device)
        y = y.to(device)
        output = model(X)

        loss += lossfn(output, y).item()
        score += scorefn(output, y)[0].item()

    model.train()
    return loss / vlen, score / vlen


def accuracy(output, target, topk=(3,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mapk(output, target, k=3):
    """
    Computes the mean average precision at k.

    Parameters
    ----------
    output (torch.Tensor): A Tensor of predicted elements.
                           Shape: (N,C)  where C = number of classes, N = batch size
    target (torch.int): A Tensor of elements that are to be predicted.
                        Shape: (N) where each value is  0≤targets[i]≤C−1
    k (int, optional): The maximum number of predicted elements

    Returns
    -------
    score (torch.float):  The mean average precision at k over the output
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        for i in range(k):
            correct[i] = correct[i] * (k - i)

        score = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        score.mul_(1.0 / (k * batch_size))
        return score

a = torch.randn(10,2,220,200)

k=5
a[:k].view(-1)

model = torchvision.models.resnet18(pretrained=True)
def squeeze_weights(m):
    m.weight.data = m.weight.data.sum(dim=1)[:,None]
    m.in_channels = 1

model.conv1.apply(squeeze_weights)

num_classes = 340

model.fc = nn.Linear(512, out_features=num_classes, bias=True)

model(torch.randn(12,1,224,224)).size()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 12000, 18000], gamma=0.5)

epochs = 1
lsize = len(train_dataloader)
print(f"size of train : {lsize}")
itr = 1
p_itr = 1000
model.train()
tloss, score = 0, 0

testset = DoodleDataset('test_simplified1.csv', 'C:/Users/User/Desktop/', mode='test', nrows=None, size=SIZE)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

import tqdm

model.eval()
model = model.to(device)
labels = np.empty((0,3))
#labels = labels.to(device)
for x in tqdm.tqdm(testloader):
    x = x.to(device)
    output = model(x)
    _, pred = output.topk(3, 1, True, True)
    labels = np.concatenate([labels, pred.cpu()], axis = 0)

submission = pd.read_csv('C:/Users/User/Desktop/test_simplified1.csv', index_col='key_id')
submission.drop(['countrycode', 'drawing'], axis=1, inplace=True)
submission['word'] = ''
for i, label in enumerate(labels):
    submission.word.iloc[i] = " ".join([decode_labels(l) for l in label])

submission.to_csv('submission1.csv')