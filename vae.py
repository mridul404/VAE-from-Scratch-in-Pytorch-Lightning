import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import lightning as L 
import torchvision.transforms as transforms 
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader 
import os  
from PIL import Image


inverse_normalize = transforms.Normalize((-1, -1, -1), (2, 2, 2))


class VAE(L.LightningModule):
    def __init__(self):
        super(VAE, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.mean = nn.Linear(16384, 16)
        self.std = nn.Linear(16384, 16)

        self.z_to_hid = nn.Linear(16, 16384)

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encode(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        mean, std = self.mean(x), self.std(x)
        z = mean + std * torch.randn_like(mean)
        # print(z.shape)
        x = self.z_to_hid(z)
        # print(x.shape)
        x = x.view(-1, 256, 8, 8)
        x = self.decode(x)
        # print(x.shape)
        return x , mean, std

    def sample_images(self, num_samples=64):
        # Generate random samples from the latent space
        z = torch.randn(num_samples, 16).to(self.device)

        # Pass through the decoder
        x = self.z_to_hid(z)
        x = x.view(-1, 256, 8, 8)
        generated_images = self.decode(x)
        
        generated_images = inverse_normalize(generated_images)


        # Display the generated images (optional)
        grid = vutils.make_grid(generated_images, nrow=8, normalize=True)
        vutils.save_image(grid, "sampled_images.png")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch
        img, mean, std = self.forward(x)
        reconstruction_loss = F.mse_loss(img, x, reduction='sum')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mean ** 2 - std**2, dim = 1), dim = 0)
        loss = reconstruction_loss + kld_loss
        self.log_dict({'train_loss': loss, 'recon_loss': reconstruction_loss, 'kld_loss': kld_loss}, prog_bar=True)
        return loss 
    
    def train_dataloader(self):
        ds = MonetDataset('./dataset')
        return DataLoader(ds, batch_size=150, shuffle=True, num_workers=2)


class MonetDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        self.images = os.listdir(image_path)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images_name = os.path.join(self.image_path, self.images[idx])
        images = Image.open(images_name)
        images = self.transform(images)
        return images

# ds = MonetDataset('./dataset')
# trainloader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
# image = iter(trainloader)
# print(next(image).shape)

# dummy = torch.rand(1, 3, 32, 32)
# model = VAE()
# output = model(dummy)

if __name__ == '__main__':
    # trainer = L.Trainer(fast_dev_run=True)
    trainer = L.Trainer(
        max_epochs=1000,
        precision='16-mixed',

    )
    vae = VAE()
    trainer.fit(vae)

    vae.sample_images()
