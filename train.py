# This will allow training Conditional VAE based on provided data
# This script is based on Pytorch
import torch
import numpy as np
from cvae import ConditionalVAE
from utils import save_output

# this helps to re-generate exact results when random values are initialized.
torch.manual_seed(32)
if torch.cuda.is_available():
    torch.cuda.manual_seed(32)


if __name__ == '__main__':
    z = torch.randn([32, 128])
    conditions = np.random.choice(range(10), size=(32,))

    for alpha in [0.2, 0.5, 0.7, 1.0]:
        model = ConditionalVAE(f'./models/alpha-{alpha}', 'cuda' if torch.cuda.is_available() else 'cpu')
        model.train(64, 200, lr=1e-3, alpha=alpha)

        model.load()
        images = model.generate(z, conditions)
        save_output(images, conditions, f'./models/alpha-{alpha}/output.png')
