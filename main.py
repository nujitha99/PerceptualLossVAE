import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFilter
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim


class BlurAndDownsample(nn.Module):
    def __init__(self, sigma=1.0, scale_factor=2):
        super(BlurAndDownsample, self).__init__()
        self.sigma = sigma
        self.scale_factor = scale_factor

    def gaussian_blur(self, img):
        # Applying Gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(self.sigma))
        return img

    def downsample(self, img):
        # Downsample using bicubic interpolation
        width, height = img.size
        new_size = (width // self.scale_factor, height // self.scale_factor)
        img = TF.resize(img, new_size, interpolation=transforms.InterpolationMode.BICUBIC)
        return img

    def forward(self, img):
        img = self.gaussian_blur(img)
        img = self.downsample(img)
        return img


def prepare_datasets(image_size=128, batch_size=32, scale_factor=2, test_split=0.2, limit_dataset_size=1000):
    # High-Resolution Transform
    hr_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Low-Resolution Transform
    lr_transform = transforms.Compose([
        transforms.Resize((image_size * scale_factor, image_size * scale_factor)),  # Resize for downsampling
        BlurAndDownsample(sigma=1.0, scale_factor=scale_factor),
        transforms.Resize((image_size, image_size)),  # Resize back to desired LR size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_dir = 'data'

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Download and load the full dataset
    full_dataset = OxfordIIITPet(root='/data', download=True, transform=None)

    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    limited_train_size = min(limit_dataset_size, train_size)
    limited_test_size = min(limit_dataset_size, test_size)

    train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                     [limited_train_size, train_size - limited_train_size])
    test_dataset, _ = torch.utils.data.random_split(test_dataset, [limited_test_size, test_size - limited_test_size])

    def lr_hr_transform(data):
        hr_image = hr_transform(data)
        lr_image = lr_transform(data)
        return lr_image, hr_image

    # Update dataset classes to use custom transform
    train_dataset = TransformDataset(train_dataset, lr_hr_transform)
    test_dataset = TransformDataset(test_dataset, lr_hr_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][0]
        return self.transform(data)


train_loader, test_loader = prepare_datasets()


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 16 * 16, 256)
        self.fc_logvar = nn.Linear(64 * 16 * 16, 256)

        # Decoder
        self.decoder_input = nn.Linear(256, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder_input(z)
        x = x.view(-1, 64, 16, 16)
        decoded = self.decoder(x)
        return decoded, mu, logvar


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_features = models.vgg16(pretrained=True).features[:16].eval()
        self.feature_extractor = nn.Sequential(*vgg_features)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        content_loss = F.mse_loss(self.feature_extractor(target), self.feature_extractor(output))
        return content_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
perceptual_loss_fn = PerceptualLoss().to(device)


def save_tensor_images(images, epoch, batch):
    output_dir = 'outputs'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = f'{output_dir}/epoch_{epoch}_batch_{batch}.png'

    save_image(images.cpu(), file_path, nrow=4)
    print(f"Saved images to {file_path}")


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor * std + mean


def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # Calculate KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.001 * MSE + KLD


models_dir = 'models'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


epochs = 100


def train_and_validate(model, train_loader, test_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(epochs):
        for i, (lr_images, hr_images) in enumerate(train_loader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            optimizer.zero_grad()
            sr_images, mu, logvar = model(lr_images)  # Generate super-resolved images and latent variables

            # Calculate losses
            recon_loss = loss_function(sr_images, hr_images, mu, logvar)
            perceptual_loss = perceptual_loss_fn(sr_images, hr_images)
            loss = recon_loss + 0.8 * perceptual_loss

            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

            sr_images = denormalize(sr_images)
            if i % 500 == 0:
                # Save training progress images
                compare_images = torch.cat([hr_images[:4], sr_images[:4]])
                save_tensor_images(compare_images, epoch, i)  # Save images to disk

        # # Validation
        # model.eval()
        # with torch.no_grad():
        #     total_loss = 0
        #     count = 0
        #     for lr_images, hr_images in test_loader:
        #         lr_images, hr_images = lr_images.to(device), hr_images.to(device)
        #
        #         reconstruction, mu, logvar = model(lr_images)
        #
        #         # Calculate losses
        #         recon_loss = loss_function(reconstruction, hr_images, mu, logvar)
        #         perceptual_loss = perceptual_loss_fn(reconstruction, hr_images)
        #         loss = recon_loss + 50 * perceptual_loss
        #         total_loss += loss.item()
        #         count += 1
        #     avg_loss = total_loss / count
        #     print(f"Epoch: {epoch}, Validation Loss: {avg_loss}")

        # Save the model
        torch.save(model.state_dict(), f'models/vae_epoch_{epoch + 1}.pth')


def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            outputs, _, _ = model(images)
            if i == 0:
                compare_images(images, outputs)


def compare_images(original, reconstructed):
    if original.shape != reconstructed.shape:
        reconstructed = TF.resize(reconstructed, original.shape[-2:])
    figure, ax = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        ax[0, i].imshow(original[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        ax[0, i].axis('off')
        ax[1, i].imshow(reconstructed[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        ax[1, i].axis('off')
    plt.show()


def calculate_psnr(model, dataloader):
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs, _, _ = model(images)
            outputs = TF.resize(outputs, images.shape[-2:])
            for i in range(images.size(0)):
                psnr = compare_psnr(images[i].cpu().numpy(), outputs[i].cpu().numpy(), data_range=1)
                total_psnr += psnr
    average_psnr = total_psnr / len(dataloader.dataset)
    print(f'Average PSNR: {average_psnr} dB')


def calculate_ssim(model, dataloader):
    model.eval()
    total_ssim = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs, _, _ = model(images)
            outputs = TF.resize(outputs, images.shape[-2:])
            for i in range(images.size(0)):
                img_np = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                output_np = (outputs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                ssim = compare_ssim(img_np, output_np, multichannel=True, win_size=3)
                total_ssim += ssim
    average_ssim = total_ssim / len(dataloader.dataset)
    print(f'Average SSIM: {average_ssim}')


train_and_validate(model, train_loader, test_loader, epochs)
test(model, test_loader)


def load_model(model_path):
    model = VAE()
    model.load_state_dict(torch.load(model_path))
    return model


model_path = f"models/vae_epoch_{epochs}.pth"
model = load_model(model_path)

calculate_psnr(model, test_loader)
calculate_ssim(model, test_loader)