# This will allow training Conditional VAE based on provided data
# This script is based on Pytorch
import math

import click
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import MNIST
import torchvision.transforms as vision_transforms

from models import ConditionalVaeFFN, idx2onehot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(32)  # this helps to re-generate exact results when random values are initialized.
if torch.cuda.is_available():
    torch.cuda.manual_seed(32)


    def loss_fn(recon_x, x, mean, log_var):
        BCE = nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)


@click.command()
def train(store_path: str = None, epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
    """Train the model using parameters provided."""
    dataset = MNIST(root='data', train=True, transform=vision_transforms.ToTensor(), download=True)
    print(f"Dataset: MNIST, Total samples found: {len(dataset)}")

    # split the dataset to train and validation sets
    validation_split = int(math.floor(len(dataset) * 0.2))
    train, validation = random_split(dataset, [len(dataset) - validation_split, validation_split])
    print(f"Training samples: {len(train)} Validation Samples: {len(validation)}")
    assert len(train) + len(validation) == len(dataset), f"Split is not proper"

    # loaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=4)

    # TODO: initialize model, optimizer
    model = ConditionalVaeFFN(28*28, 256, 128, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    results = {}
    for epoch in range(epochs):
        epoch_details = {}
        losses = []
        for idx, (x, y) in enumerate(train_loader):
            x = x.view(-1, 28*28)
            y = y.view(-1, 1).type(torch.int64)
            x, y = x.to(device), y.to(device)
            y = idx2onehot(y, 10)

            # Forward pass
            recon_x, mean, var, z = model(x, y)

            loss = loss_fn(recon_x, x, mean, var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if idx % 500 == 1:
                print(f"Epoch ({epoch}) - batch ({idx}) loss: {losses[-1]}")

        print(f"Epoch-{epoch} train-loss: {np.average(losses)}")


if __name__ == '__main__':
    train()

