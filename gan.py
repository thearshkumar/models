"""
Credit for guidance: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#training
"""
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim = 32 * 3):
        super().__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim // 2, kernel_size = 4, bias = False),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(latent_dim // 2, latent_dim // 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(latent_dim // 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(latent_dim // 4, latent_dim // 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(latent_dim // 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(latent_dim // 8, latent_dim // 16, 4, 2, 1, bias = False),
            nn.BatchNorm2d(latent_dim // 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(latent_dim // 16, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.unsqueeze(2).unsqueeze(3)
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(out_channels, out_channels * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(out_channels * 2, out_channels * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(out_channels * 4),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(out_channels * 4, out_channels * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(out_channels * 8),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(out_channels * 8, 1, 4, 1, 0, bias = False)
        )
    
    def forward(self, x):
        return self.model(x)
    
def train(train_dataloader, latent_dim = 48, batch_size = 32, epochs = 100):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    g_model = Generator().to(device)
    d_model = Discriminator().to(device)

    disc_loss_fn = nn.BCEWithLogitsLoss()
    d_opt = torch.optim.Adam(d_model.parameters(), lr = 1e-3, betas = (0.5, 0.999))
    g_opt = torch.optim.Adam(g_model.parameters(), lr = 1e-3, betas = (0.5, 0.999))

    for epoch in range(epochs):
        for i, img in enumerate(train_dataloader):
            img = img.to(device)

            # Discriminator Training

            d_opt.zero_grad()

            # Real images

            real_output = d_model(img)
            d_loss_real = disc_loss_fn(real_output, torch.ones_like(real_output))

            # Fake images

            z = torch.randn((batch_size, latent_dim), device = device)
            fake_images = g_model(z)
            fake_output = d_model(fake_images.detach())
            d_loss_fake = disc_loss_fn(fake_output, torch.zeros_like(fake_output))
            
            d_total_loss = d_loss_real + d_loss_fake
            d_total_loss.backward()
            d_opt.step()

            # Generator Training

            d_fake_output = d_model(fake_images)

            g_opt.zero_grad()
            g_loss = disc_loss_fn(d_fake_output, torch.ones_like(d_fake_output))
            g_loss.backward()
            g_opt.step()
    
    return g_model, d_model