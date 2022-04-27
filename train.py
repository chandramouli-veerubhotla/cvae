# This will allow training Conditional VAE based on provided data
# This script is based on Pytorch
from typing import List
import math
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import MNIST
import torchvision.transforms as vision_transforms

import json
from models import ConditionalVaeFFN
from utils import label_onehot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(32)  # this helps to re-generate exact results when random values are initialized.
if torch.cuda.is_available():
    torch.cuda.manual_seed(32)


class ConditionalVAE:
    """Generate handwritten digit images based on provided condition using Conditional VAE.

    """
    def __init__(self, model_dir: str, device: str = 'cpu'):
        self.dir = os.path.abspath(model_dir)

        self.device = torch.device(device)
        self.model = ConditionalVaeFFN(28*28, 256, 128, 10).to(self.device)

    def load(self):
        model_path = os.path.join(self.dir, 'cvae.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

        raise NotImplementedError

    def save(self, training_info: dict):
        os.makedirs(self.dir, exist_ok=True)
        json.dump(training_info, open(f'{self.dir}/training_info.json', 'w+'))

        model_path = os.path.join(self.dir, 'cvae.pth')
        torch.save(self.model.state_dict(), model_path)

    def _reconstruction_loss(self, x, x_hat, dimensions):
        return F.binary_cross_entropy(x_hat.view(-1, dimensions), x.view(-1, dimensions), reduction='sum')

    def _kl_distance(self, mean, var):
        return -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())

    def _total_loss(self, recon_x, x, mean, var):
        rec = self._reconstruction_loss(x, recon_x, 28 * 28)
        kl = self._kl_distance(mean, var)

        return (rec + kl) / x.size(0)

    def _train_step(self, loader, optimizer, print_every: int = 500):
        train_losses = []
        self.model.train()
        for idx, (x, y) in enumerate(loader):
            x = x.view(-1, 28 * 28)
            y = y.view(-1, 1).type(torch.int64)
            x, y = x.to(self.device), y.to(self.device)
            y = label_onehot(y, 10)

            # Forward pass
            x_hat, mean, var, z = self.model(x, y)

            loss = self._total_loss(x_hat, x, mean, var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if idx % print_every == 1:
                print(f"\tBatch {idx}: loss: {train_losses[-1]:.3f}")

        return np.average(train_losses)

    def _validation_step(self, loader, print_every: int = 500):
        val_losses = []
        self.model.eval()
        for idx, (x, y) in enumerate(loader):
            x = x.view(-1, 28 * 28)
            y = y.view(-1, 1).type(torch.int64)
            x, y = x.to(self.device), y.to(self.device)
            y = label_onehot(y, 10)

            x_hat = self.model.generate(y)
            batch_loss = self._reconstruction_loss(x, x_hat, 28 * 28) / x.size(0)
            val_losses.append(batch_loss.item())

            if idx % print_every == 1:
                print(f"\tBatch {idx}: loss: {val_losses[-1]:.3f}")

        return np.average(val_losses)

    def train(self, batch_size: int, max_epochs: int, lr=1e-3):
        dataset = MNIST(root='data', train=True, transform=vision_transforms.ToTensor(), download=True)
        print(f"Dataset: MNIST, Total samples found: {len(dataset)}")

        # split the dataset to train and validation sets
        validation_split = int(math.floor(len(dataset) * 0.2))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - validation_split, validation_split])
        print(f"Training samples: {len(train_dataset)} Validation Samples: {len(val_dataset)}")
        assert len(train_dataset) + len(val_dataset) == len(dataset), f"Split is not proper"

        # loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        training_info = {}

        for epoch in range(max_epochs):
            train_loss = self._train_step(train_loader, optimizer, print_every=500)
            val_loss = self._validation_step(val_loader, print_every=500)

            print(f"---------------------------------------------")
            print(f"Epoch ({epoch}/{max_epochs}): train_loss: {train_loss:.3f} validation_loss: {val_loss:.3f}")

            training_info[epoch] = {
                'train_loss': train_loss,
                'validation_loss': val_loss
            }

            # TODO: save only best
            # TODO: calculate metric
            self.save(training_info)
            print(f"Saved best model, ")

    def generate(self, conditions: List[int]):
        conditions = torch.IntTensor(conditions).to(self.device)
        conditions = label_onehot(conditions, 10)

        self.model.eval()
        images = self.model.generate(conditions)
        return images


if __name__ == '__main__':
    model = ConditionalVAE('model', 'cuda')
    model.train(32, 20, lr=1e-3)

