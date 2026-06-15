import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from algo_v2.config import CHECKPOINTS_DIR, PRIMARY_DATA_PATH

CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "phase1_gan")
MODEL_PATH = os.path.join(CKPT_DIR, "market_generator.pth")
PARAMS_PATH = os.path.join(CKPT_DIR, "gan_scaling_params.csv")

# Configuration
SEQ_LEN = 24
BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 2000
N_CRITIC = 5
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(f"Training on device: {device}")


# 1. Data Preparation
def load_and_preprocess_data():
    df = pd.read_csv(PRIMARY_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "tic"])

    wide_df = df.pivot(index="date", columns="tic", values="close")
    wide_df = wide_df.ffill().bfill().dropna()

    returns_df = wide_df.pct_change().dropna()

    mean = returns_df.mean()
    std = returns_df.std()
    returns_normalized = (returns_df - mean) / std

    os.makedirs(CKPT_DIR, exist_ok=True)
    pd.DataFrame({"mean": mean, "std": std}).to_csv(PARAMS_PATH)

    return returns_normalized.values, len(wide_df.columns)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx : idx + self.seq_len])


# 2. GAN Architecture
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z shape: (batch_size, seq_len, latent_dim)
        h0 = torch.zeros(self.num_layers, z.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, z.size(0), self.hidden_dim).to(device)

        out, _ = self.lstm(z, (h0, c0))
        out = self.linear(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        out, _ = self.lstm(x, (h0, c0))
        # Use only the last time step for sequence classification
        out = self.linear(out[:, -1, :])
        return out


# 3. Gradient Penalty for WGAN-GP
def calculate_gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1).to(device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.requires_grad_(True)

    with torch.backends.cudnn.flags(enabled=False):
        d_interpolates = discriminator(interpolates)

    fake = torch.ones(batch_size, 1).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# 4. Training Loop
def train_wgan_gp():
    print("Loading data...")
    data, num_features = load_and_preprocess_data()
    print(f"Data shape: {data.shape}. Number of assets (features): {num_features}")

    dataset = TimeSeriesDataset(data, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Initialize models
    netG = Generator(LATENT_DIM, 64, num_features).to(device)
    netD = Discriminator(num_features, 64).to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9))

    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # --- Train Discriminator ---
            for _ in range(N_CRITIC):
                netD.zero_grad()

                # Real data loss
                d_real = netD(real_data)
                loss_d_real = -torch.mean(d_real)

                # Fake data loss
                z = torch.randn(batch_size, SEQ_LEN, LATENT_DIM).to(device)
                fake_data = netG(z)
                d_fake = netD(fake_data)
                loss_d_fake = torch.mean(d_fake)

                # Gradient Penalty
                gradient_penalty = calculate_gradient_penalty(netD, real_data.data, fake_data.data)

                # Total Discriminator Loss
                loss_d = loss_d_real + loss_d_fake + 10.0 * gradient_penalty
                loss_d.backward()
                optimizerD.step()

            # --- Train Generator ---
            netG.zero_grad()
            z = torch.randn(batch_size, SEQ_LEN, LATENT_DIM).to(device)
            fake_data = netG(z)
            d_fake = netD(fake_data)

            # Generator wants Discriminator to output large positive values
            loss_g = -torch.mean(d_fake)
            loss_g.backward()
            optimizerG.step()

        if epoch % 10 == 0:
            print(f"[{epoch}/{EPOCHS}] Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(netG.state_dict(), MODEL_PATH)
    print(f"Training complete! Generator saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_wgan_gp()
