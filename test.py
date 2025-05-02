import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import AutoEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder().to(device)
model.load_state_dict(torch.load('AE.pth',weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])


def visualize_test(model, device, test_loader):
    with torch.no_grad():
        for data in test_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
            output = model(img)
            output = output.view(output.size(0), 1, 28, 28)
            img = img.view(img.size(0), 1, 28, 28)

            comparison = torch.cat([img[:5], output[:5]])
            comparison = comparison.cpu()
            plt.figure(figsize=(10, 2))
            for i in range(10):
                plt.subplot(1, 10, i + 1)
                plt.imshow(comparison[i].squeeze().numpy(), cmap='gray')
                plt.axis('off')
            plt.show()
            break


test_dataset = datasets.MNIST(root='./', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

visualize_test(model, device, test_loader)
