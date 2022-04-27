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
    plt.figure(figsize=(5, 10))

    # iterate over each image and add each image as subplot to main figure
    for idx, (image, condition) in enumerate(zip(images, conditions)):
        if idx > 9:
            break
        plt.subplot(5, 2, idx+1)

        # Add condition information to reference
        plt.text(0, 0, "label={:d}".format(condition.item()), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image.view(28, 28).cpu().data.numpy())
        plt.axis('off')

    # Save generated image to desired location
    plt.savefig(save_to, dpi=300)

    # clean and close all matplotlib stuff
    plt.clf()
    plt.close('all')


# Refer this https://keras.io/examples/generative/vae/ to generate model images
