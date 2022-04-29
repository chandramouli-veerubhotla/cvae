import torch
import numpy as np
from cvae import ConditionalVAE
from utils import save_output

# this helps to re-generate exact results when random values are initialized.
torch.manual_seed(32)
if torch.cuda.is_available():
    torch.cuda.manual_seed(32)


if __name__ == '__main__':
    model = ConditionalVAE('model', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.load()

    z = torch.randn([32, 128])
    conditions = np.random.choice(range(10), size=(32, 1))
    images = model.generate(z, conditions)
    save_output(images, conditions, 'model-1-output.png')

