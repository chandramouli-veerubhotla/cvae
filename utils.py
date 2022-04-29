import numpy as np
import torch
import matplotlib.pyplot as plt


def label_onehot(idx, n):
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot


def save_output(images, conditions, save_to: str):
    total_images = images.size(0)
    rows = total_images // 10
    # iterate over each image and add each image as subplot to main figure
    for idx, (image, condition) in enumerate(zip(images, conditions)):
        plt.subplot(rows+1, 10, idx+1)

        # Add condition information to reference
        plt.text(0, 0, "label={:d}".format(condition), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image.view(28, 28).cpu().data.numpy())
        plt.axis('off')

    # Save generated image to desired location
    plt.savefig(save_to, dpi=300)

    # clean and close all matplotlib stuff
    plt.clf()
    plt.close('all')


def plot_loss(data, save_to: str):
    train_loss, val_loss = [], []

    best = data['best']
    del data['best']

    for epoch, losses in data.items():
        train_loss.append(losses['train_loss'])
        val_loss.append(losses['validation_loss'])

    plt.figure(figsize=(5, 5))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.vlines(x = int(best), color = 'b', ymin=min(train_loss[best], val_loss[best]) - 0.1, ymax=max(train_loss[best], val_loss[best]) + 0.1, label = 'best model saved')

    plt.title('Train vs Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # Save generated image to desired location
    plt.savefig(save_to, dpi=300)

    # clean and close all matplotlib stuff
    plt.clf()
    plt.close('all')
