import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channels, 32 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

        self.dropout1 = nn.Dropout(p=0.2, inplace=False)

    # backward function already defined through autograd
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)

        x = x.view(-1, 128 * 2 * 2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)  # output layer

        return x


def create_model(device):
    # define network
    model = Net()
    print(model)
    model.to(device)

    return model


def load_model(model):
    path = './cifar_net.pth'
    model.load_state_dict(torch.load(path))
    model.eval()


def save_model(model):
    path = './cifar_net.pth'
    torch.save(model.state_dict(), path)
