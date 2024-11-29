from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import time


class ConvFiltersTransform:
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, img):
        # Convert the PyTorch tensor to a NumPy array
        img_np = img.numpy()

        # Apply the Conv_filters function
        filtered_img_np = Conv_filters(img_np, axis=self.axis)

        # Convert the result back to a PyTorch tensor
        filtered_img_tensor = torch.tensor(filtered_img_np, dtype=torch.float32)
        # print('slow')
        return filtered_img_tensor


def train():
    # TRANSFORMATION AUGMENTATION
    data_transform = transforms.Compose([
        transforms.Resize(size=(14, 14)),
        # Turn the image into a torch.Tensor
        transforms.ToTensor(),  # converts all pixel values from 0 to 255 to be between 0.0 and 1.0
        ConvFiltersTransform(axis=(1, 2)),
        transforms.Resize(size=(224, 224))
    ])

    # DATA SETS
    train_data = datasets.ImageFolder(root='./Dataset/train', transform=data_transform)
    test_data = datasets.ImageFolder(root='./Dataset/test', transform=data_transform)

    # DATA LOADER
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

    # print(train_data[0][0].shape)  # C, H, W

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv7 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=7, stride=1, padding=3)
            self.conv5_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=3, padding=3)
            self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=3, padding=5)

            self.conv3_1_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
            self.conv3_1_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
            self.conv3_1_3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
            self.conv3_1_4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)

            self.conv5_2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
            self.conv5_2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
            self.conv5_2_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
            self.conv5_2_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)

            self.conv3_3_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
            self.conv3_3_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
            self.conv3_3_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
            self.conv3_3_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

            self.conv5_4_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
            self.conv5_4_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
            self.conv5_4_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
            self.conv5_4_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)

            self.avgpool_1 = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1)
            self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)

            self.conv4 = nn.Conv2d(in_channels=128, out_channels=750, kernel_size=3, padding=2)
            self.avgpool = nn.AvgPool2d(kernel_size=3, padding=0, stride=2)

            self.fc1 = nn.Linear(750 * 4 * 4, 750)
            self.fc2 = nn.Linear(750, 3)

        def forward(self, x):
            x_7 = F.relu(self.conv7(x))
            x_5_1 = F.relu(self.conv5_1(x_7))
            x_5_2 = F.relu(self.conv5_2(x_5_1))
            x_5_2 = nn.BatchNorm2d(32)(x_5_2)

            x_3_1_1 = F.relu(self.conv3_1_1(x_5_2))
            x_3_1_2 = F.relu(self.conv3_1_2(x_5_2))
            x_3_1_3 = F.relu(self.conv3_1_3(x_5_2))
            x_3_1_4 = F.relu(self.conv3_1_4(x_5_2))

            x_5_2_1 = F.relu(self.conv5_2_1(x_3_1_1))
            x_5_2_2 = F.relu(self.conv5_2_2(x_3_1_2))
            x_5_2_3 = F.relu(self.conv5_2_3(x_3_1_3))
            x_5_2_4 = F.relu(self.conv5_2_4(x_3_1_4))

            x_3_3_1 = F.relu(self.conv3_3_1(x_5_2_1))
            x_3_3_2 = F.relu(self.conv3_3_2(x_5_2_2))
            x_3_3_3 = F.relu(self.conv3_3_3(x_5_2_3))
            x_3_3_4 = F.relu(self.conv3_3_4(x_5_2_4))

            x_5_4_1 = F.relu(self.conv5_4_1(x_3_3_1))
            x_5_4_2 = F.relu(self.conv5_4_2(x_3_3_2))
            x_5_4_3 = F.relu(self.conv5_4_3(x_3_3_3))
            x_5_4_4 = F.relu(self.conv5_4_4(x_3_3_4))


            # print(x2_1.shape)
            x_cat_1 = torch.cat((x_5_4_1, x_5_4_2), 1)
            x_cat_2 = torch.cat((x_5_4_3, x_5_4_4), 1)
            x_sum_1 = F.relu(torch.add(x_cat_1, x_cat_2))
            x_sum_2 = F.relu(torch.add(x_sum_1, x_5_2))

            x_avgpool_1 = F.relu(self.avgpool_1(x_sum_2))
            x_conv3 = F.relu(self.conv3(x_avgpool_1))
            x_maxpool = F.relu(self.maxpool(x_conv3))
            # print(x_maxpool.shape)

            x_conv4 = F.relu(self.conv4(x_maxpool))
            x_avgpool = F.relu(self.avgpool(x_conv4))

            # print(x_avgpool.shape)

            x_fc = x_avgpool.view(-1, 750 * 4 * 4)
            x_fc = F.relu(self.fc1(x_fc))
            x_fc = self.fc2(x_fc)
            return x_fc

    net = Net()

    summary(net, input_size=(12, 224, 224))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        time_start = time.time()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
                print('Time:', time.time() - time_start)
                time_start = time.time()

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    train()
