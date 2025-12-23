"""
Copied from article by Mostafa Wael(link: https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3/)

Implementation is rudimentary and long for clarity. There is a lot of scope for 
reduction in code size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_class = 1000, in_channels = 3):
        super().__init__()

        # contracting path
        # econv = encoder conv layer
        self.econv1_1 = nn.Conv2d(in_channels, 64, kernel_size = 3, padding = 1)
        self.econv1_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.pool1    = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.econv2_1 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.econv2_2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.pool2    = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.econv3_1 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.econv3_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.pool3    = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.econv4_1 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
        self.econv4_2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.pool4    = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # middle
        self.econv5_1  = nn.Conv2d(512, 1024, kernel_size = 3, padding = 1)
        self.econv5_2  = nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1)

        # expansion path

        self.upconv1   = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.dconv1_1 = nn.Conv2d(1024, 512, kernel_size = 3, padding = 1)
        self.dconv1_2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)

        self.upconv2   = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.dconv2_1 = nn.Conv2d(512, 256, kernel_size = 3, padding = 1)
        self.dconv2_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)

        self.upconv3   = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.dconv3_1 = nn.Conv2d(256, 128, kernel_size = 3, padding = 1)
        self.dconv3_2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        
        self.upconv4   = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.dconv4_1 = nn.Conv2d(128, 64, kernel_size = 3, padding = 1)
        self.dconv4_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)

        self.outconv   = nn.Conv2d(64, n_class, kernel_size = 1)

    def forward(self, x):

        # Encoder
        x_e_1_1 = F.relu(self.econv1_1(x))
        x_e_1_2 = F.relu(self.econv1_2(x_e_1_1))
        x_p_1 = self.pool1(x_e_1_2)

        x_e_2_1 = F.relu(self.econv2_1(x_p_1))
        x_e_2_2 = F.relu(self.econv2_2(x_e_2_1))
        x_p_2 = self.pool2(x_e_2_2)

        x_e_3_1 = F.relu(self.econv3_1(x_p_2))
        x_e_3_2 = F.relu(self.econv3_2(x_e_3_1))
        x_p_3 = self.pool3(x_e_3_2)
        
        x_e_4_1 = F.relu(self.econv4_1(x_p_3))
        x_e_4_2 = F.relu(self.econv4_2(x_e_4_1))
        x_p_4 = self.pool4(x_e_4_2)

        x_e_5_1 = F.relu(self.econv5_1(x_p_4))
        x_e_5_2 = F.relu(self.econv5_2(x_e_5_1))

        # Decoder

        x_up_1 = self.upconv1(x_e_5_2)
        # Res connection
        x_up_1_cat = torch.cat([x_up_1, x_e_4_2], dim = 1)
        x_d_1_1 = F.relu(self.dconv1_1(x_up_1_cat))
        x_d_1_2 = F.relu(self.dconv1_2(x_d_1_1))

        x_up_2 = self.upconv2(x_d_1_2)
        # Res connection
        x_up_2_cat = torch.cat([x_up_2, x_e_3_2], dim = 1)
        x_d_2_1 = F.relu(self.dconv2_1(x_up_2_cat))
        x_d_2_2 = F.relu(self.dconv2_2(x_d_2_1))

        x_up_3 = self.upconv3(x_d_2_2)
        # Res connection
        x_up_3_cat = torch.cat([x_up_3, x_e_2_2], dim = 1)
        x_d_3_1 = F.relu(self.dconv3_1(x_up_3_cat))
        x_d_3_2 = F.relu(self.dconv3_2(x_d_3_1))

        x_up_4 = self.upconv4(x_d_3_2)
        # Res connection
        x_up_4_cat = torch.cat([x_up_4, x_e_1_2], dim = 1)
        x_d_4_1 = F.relu(self.dconv4_1(x_up_4_cat))
        x_d_4_2 = F.relu(self.dconv4_2(x_d_4_1))

        # Output layer
        output = self.outconv(x_d_4_2)

        return output

# For testing
# if __name__ == "__main__":
#     rand_img = torch.randint(low = 0, high = 255, size = (32, 3, 128, 128), dtype = torch.float32)
#     model = UNet()
#     output = model(rand_img)
#     print(output)