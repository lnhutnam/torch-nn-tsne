import os
import random

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import numpy as np
import pandas as pd

from model import ResNet
from layers import BasicBlock
from utils import train_val_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 2024

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set as {seed}")

num_epochs = 5
num_classes = 10
batch_size = 128
learning_rate = 0.001

torch.manual_seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = torchvision.datasets.MNIST(
    "data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST("data", train=False, transform=transform)

# Before
print("Train data set:", len(train_dataset))
print("Test data set:", len(test_dataset))

# Random split
train_set_size = int(len(train_dataset) * 0.8)
indices = list(range(train_set_size))
split = int(np.floor(0.2 * train_set_size))
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SequentialSampler(val_indices)


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=train_sampler,
    batch_size=batch_size,
)

valid_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=valid_sampler,
    batch_size=batch_size,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False
)

# ResNet18
# model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=number of classes you want to classify)

# ResNet34
# model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=number of classes you want to classify)

# ResNet50
# model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=number of classes you want to classify)

# ResNet101
# model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=number of classes you want to classify)

# ResNet150
# model = ResNet(Bottleneck, [3, 6, 36, 3], num_classes=number of classes you want to classify)
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

dataloaders = dict(train=train_loader, val=valid_loader)
criterion = torch.nn.CrossEntropyLoss()
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print(model)
print(
    "Total number of parameters =",
    np.sum([np.prod(parameter.shape) for parameter in model.parameters()]),
)
model, losses, accuracies = train_val_model(
    model, device, criterion, optimizer, dataloaders, num_epochs=1, log_interval=1
)

# _ = plt.plot(losses['train'], '-b', losses['val'], '--r')

count = 0
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if count % 100 == 0:
            labels = labels.cpu().numpy().reshape(-1, 1)
            avgpool_output = model.conv1(images).view(images.size(0), -1).cpu().numpy()
            df_features = pd.DataFrame(avgpool_output)
            df_features.to_csv("vis.tsv", mode="a", header=False, sep="\t", index=False)
            df_labels = pd.DataFrame(labels)
            df_labels.to_csv(
                "metadata.tsv", mode="a", header=False, sep="\t", index=False
            )

        count += 1

    print(
        "Accuracy of the network on the 10000 test images: {:.5f} ".format(
            correct / total
        )
    )
