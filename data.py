from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

images, labels = next(iter(data_loader))
print(images.shape, labels)
plt.imshow(images[0].reshape(28,28).numpy())