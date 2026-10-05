from dataclasses import dataclass

import numpy as np


def getitems(l, idx):
    return [l[i] for i in idx]


@dataclass
class TrainArgs:
    n_epochs: int = 100
    min_lr: float = 1e-5
    max_lr: float = 1e-3


class DataLoader:
    def __init__(self, X, y, batch_size=16):
        self.X = X
        self.y = y
        self.size = len(X)
        self.batch_size = batch_size
        self.reshuffle()

    def reshuffle(self):
        self.perm = np.random.permutation(self.size)

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        if len(self) <= idx:
            raise IndexError
        batch_begin = self.batch_size * idx
        batch = self.perm[batch_begin:batch_begin + self.batch_size]
        return getitems(self.X, batch), getitems(self.y, batch)


def max_margin_loss(pred, target):
    size = float(len(pred))
    return sum((1.0 - pi * ti).relu() for pi, ti in zip(pred, target)) / size


class LinearOptim:
    def __init__(self, args, params):
        self.min_lr = args.min_lr
        self.max_lr = args.max_lr
        self.n_epochs = args.n_epochs
        self.params = list(params)

    def step(self, epoch):
        coef = epoch / self.n_epochs
        lr = self.max_lr + (self.min_lr - self.max_lr) * coef
        for p in self.params:
            p.data -= lr * p.grad


def train(args, model, data_loader, loss_fn):
    optim = LinearOptim(args, model.parameters())
    for i in range(args.n_epochs):
        total_loss = 0
        for batch_X, batch_y in data_loader:
            y_pred = [model(X)[0] for X in batch_X]
            loss = loss_fn(y_pred, batch_y)
            model.zero_grad()
            loss.backward()
            optim.step(i + 1)
            total_loss += loss.data
        data_loader.reshuffle()
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {i + 1} / {args.n_epochs}: loss = {avg_loss}')
