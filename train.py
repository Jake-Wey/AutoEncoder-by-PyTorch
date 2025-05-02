import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import AutoEncoder

batch_size = 64
learning_rate = 1e-3
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(root='./', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1).to(device)

        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

    print(f'Epoch[{epoch}/{epochs}], Loss: {running_loss / len(train_loader): .4f}')
print("train END!")

torch.save(model.state_dict(), 'AE.pth')
