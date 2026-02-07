"""
Credit for guidance: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
"""
from torch import nn
import torch

class CNN(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 64, img_dim = (32, 32), num_classes = 10):
        super().__init__()

        img_h, img_w = img_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(out_channels, out_channels * 2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(img_h // 16 * img_w // 16 * out_channels * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class CNN_selfpool(nn.Module):
    """
    Idea adapted from DCGAN(tutorial: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#discriminator) 
    of self-learning pooling using the 4(k_size), 2(stride), 1(padding) config
    """
    def __init__(self, in_channels = 3, out_channels = 64, img_dim = (32, 32), num_classes = 10):
        super().__init__()

        img_h, img_w = img_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(out_channels, out_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(out_channels * 2, out_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(out_channels * 4),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(out_channels * 4, out_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(out_channels * 8),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Flatten(),
            nn.Linear(img_h // 16 * img_w // 16 * out_channels * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)