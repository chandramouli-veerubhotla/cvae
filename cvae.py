from typing import List
import math
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from torchsummary import summary
from torchvision.datasets import MNIST
import torchvision.transforms as vision_transforms

import json
from models import ConditionalVaeFFN
from utils import label_onehot, save_output, plot_loss


class ConditionalVAE:
    """Generate handwritten digit images based on provided condition using Conditional VAE.

    """
    def __init__(self, model_dir: str, device: str = 'cpu'):
        self.dir = os.path.abspath(model_dir)

        self.device = torch.device(device)
        self.model = ConditionalVaeFFN(28*28, 256, 128, 10).to(self.device)
        summary(self.model, [(1, 784), (1, 10)], batch_size=32)

    def load(self):
        model_path = os.path.join(self.dir, 'cvae.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"loaded model successfully!")
        else:
            raise NotImplementedError

    def save(self, training_info: dict):
        model_path = os.path.join(self.dir, 'cvae.pth')
        torch.save(self.model.state_dict(), model_path)

    def loss(self, x_hat, x, mean, var, alpha=0.5):
        BCE = torch.nn.functional.binary_cross_entropy(
            x_hat.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
        KLD = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())

        return ((alpha * BCE) +  ((1-alpha) * KLD)) / x.size(0)

    def _train_step(self, loader, optimizer, print_every: int = 500, alpha = 0.5):
        os.makedirs(self.dir, exist_ok=True)
        train_losses = []
        self.model.train()
        for idx, (x, y) in enumerate(loader):
            x = x.view(-1, 28 * 28)
            y = y.view(-1, 1).type(torch.int64)
            x, y = x.to(self.device), y.to(self.device)
            y = label_onehot(y, 10)

            # Forward pass
            x_hat, mean, var, z = self.model(x, y)

            loss = self.loss(x_hat, x, mean, var, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if idx % print_every == 1:
                print(f"\tTrain Batch {idx}: train loss: {train_losses[-1]:.3f}")

        return np.average(train_losses)

    def _validation_step(self, loader, print_every: int = 500, alpha=0.5):
        val_losses = []
        self.model.eval()
        for idx, (x, y) in enumerate(loader):
            x = x.view(-1, 28 * 28)
            y = y.view(-1, 1).type(torch.int64)
            x, y = x.to(self.device), y.to(self.device)
            y = label_onehot(y, 10)

            x_hat, mean, var, z = self.model(x, y)
            batch_loss = self.loss(x_hat, x, mean, var, alpha=alpha)
            val_losses.append(batch_loss.item())

            if idx % print_every == 1:
                print(f"\tValidation Batch {idx}: validation loss: {val_losses[-1]:.3f}")

        return np.average(val_losses)

    def train(self, batch_size: int =32, max_epochs: int = 20, lr=1e-3, alpha=0.5, epsilon = 0.01, max_patience: int = 10):
        dataset = MNIST(root='data', train=True, transform=vision_transforms.ToTensor(), download=True)
        print(f"Dataset: MNIST, Total samples found: {len(dataset)}")

        # split the dataset to train and validation sets
        validation_split = int(math.floor(len(dataset) * 0.2))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - validation_split, validation_split])
        print(f"Training samples: {len(train_dataset)} Validation Samples: {len(val_dataset)}")
        assert len(train_dataset) + len(val_dataset) == len(dataset), f"Split is not proper"

        # loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        training_info = {}
        best_val_loss, current_patience = 99999, 0
        for epoch in range(max_epochs):
            train_loss = self._train_step(train_loader, optimizer, print_every=500, alpha=alpha)
            val_loss = self._validation_step(val_loader, print_every=500, alpha=alpha)

            print(f"Epoch ({epoch}/{max_epochs}): train_loss: {train_loss:.3f} validation_loss: {val_loss:.3f}")
            print(f"---------------------------------------------")

            training_info[epoch] = {
                'train_loss': train_loss,
                'validation_loss': val_loss
            }

            if train_loss <= 0 or val_loss <= 0:
                print(f"Stopping as loss became zero....")
                break

            if val_loss <= best_val_loss + epsilon:
                current_patience = 0
                best_val_loss = val_loss
                print(f"Saved best model when validation loss: {val_loss:.3f}")

                training_info['best'] = epoch
                self.save(training_info)
            else:
                current_patience += 1

                if current_patience >= max_patience:
                    print(f"Not finding any good learning... hence stopping")
                    break

        json.dump(training_info, open(f'{self.dir}/training_info.json', 'w+'))
        plot_loss(training_info, f"{self.dir}/loss.png")
        print(f"Finished training the model, model files can be found at {self.dir}")

    def generate(self, z, conditions):
        z = z.to(self.device)
        conditions = torch.Tensor(conditions).type(torch.int64).to(self.device)
        conditions = label_onehot(conditions, 10)

        self.model.eval()
        images = self.model.generate(z, conditions)
        return images
