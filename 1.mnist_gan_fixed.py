#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST GAN Training Script (Fixed)
---------------------------------
- Generator가 정확히 28x28을 출력하도록 수정 (1x1→7x7→14x14→28x28).
- Discriminator는 28→14→7→1 흐름(마지막 7x7 conv)으로 일치.
- 손실 곡선/샘플 이미지를 저장.
- --debug-shapes로 첫 배치에서 텐서 크기 확인.

Usage:
  python mnist_gan_fixed.py --epochs 20 --batch-size 128 --sample-every 1 --out-dir runs/mnist_gan
"""
import os
import random
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def denorm(x: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1] for visualization."""
    return (x + 1) / 2


# ----------------------------
# Models
# ----------------------------
class Generator(nn.Module):
    """
    Generator: latent_dim -> 1x28x28
    1x1 -> 7x7 -> 14x14 -> 28x28 -> 28x28(3x3 conv)
    """
    def __init__(self, latent_dim: int = 100, ngf: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # (N, latent_dim, 1, 1) -> (N, ngf*4, 7, 7)
            nn.ConvTranspose2d(latent_dim, ngf * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 7x7 -> 14x14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 28x28 유지, 채널만 1로
            nn.Conv2d(ngf, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),  # [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """
    Discriminator: 1x28x28 -> 1x1 logits
    28x28 -> 14x14 -> 7x7 -> 1x1(7x7 conv)
    """
    def __init__(self, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 28->14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 14->7
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),  # 7->1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).view(x.size(0), -1)
        return out  # (N, 1) logits


# ----------------------------
# Training
# ----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"[INFO] Using device: {device}")

    ensure_dir(args.out_dir)
    img_dir = os.path.join(args.out_dir, 'samples'); ensure_dir(img_dir)
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints'); ensure_dir(ckpt_dir)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ])
    dataset = datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)

    # Models
    netG = Generator(latent_dim=args.latent_dim, ngf=args.ngf).to(device)
    netD = Discriminator(ndf=args.ndf).to(device)

    # Loss / Opt
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label_val = 1.0
    fake_label_val = 0.0

    fixed_noise = torch.randn(args.n_sample_grid ** 2, args.latent_dim, 1, 1, device=device)

    G_losses_iter: List[float] = []
    D_losses_iter: List[float] = []
    G_losses_epoch: List[float] = []
    D_losses_epoch: List[float] = []

    first_batch_done = False

    for epoch in range(1, args.epochs + 1):
        netG.train(); netD.train()
        epoch_g_loss = 0.0; epoch_d_loss = 0.0; num_batches = 0

        for real, _ in dataloader:
            real = real.to(device)
            bsz = real.size(0)
            num_batches += 1

            # (선택) 첫 배치에서 크기 디버깅
            if args.debug_shapes and not first_batch_done:
                print(f"[DEBUG] real: {tuple(real.shape)}")  # (B,1,28,28)

            # --------- Train D ---------
            netD.zero_grad(set_to_none=True)
            labels_real = torch.full((bsz, 1), real_label_val, dtype=real.dtype, device=device)
            logits_real = netD(real)
            loss_real = criterion(logits_real, labels_real)

            noise = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
            fake = netG(noise).detach()

            # (선택) 첫 배치에서 크기 디버깅
            if args.debug_shapes and not first_batch_done:
                print(f"[DEBUG] fake (detach): {tuple(fake.shape)}")  # 기대 (B,1,28,28)
                first_batch_done = True

            # 안전 확인(에러를 조기 포착)
            assert fake.shape[2:] == (28, 28), f"Generator output is {fake.shape[2:]}, expected (28,28)"

            labels_fake = torch.full((bsz, 1), fake_label_val, dtype=real.dtype, device=device)
            logits_fake = netD(fake)
            loss_fake = criterion(logits_fake, labels_fake)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizerD.step()

            # --------- Train G ---------
            netG.zero_grad(set_to_none=True)
            noise2 = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
            gen = netG(noise2)
            assert gen.shape[2:] == (28, 28), f"Generator output is {gen.shape[2:]}, expected (28,28)"
            logits_gen = netD(gen)
            labels_gen = torch.full((bsz, 1), real_label_val, dtype=real.dtype, device=device)  # trick D
            loss_g = criterion(logits_gen, labels_gen)
            loss_g.backward()
            optimizerG.step()

            # Logs
            G_losses_iter.append(loss_g.item())
            D_losses_iter.append(loss_d.item())
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()

        # Epoch avg
        G_losses_epoch.append(epoch_g_loss / max(1, num_batches))
        D_losses_epoch.append(epoch_d_loss / max(1, num_batches))

        # Save samples
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            netG.eval()
            with torch.no_grad():
                fake_fixed = netG(fixed_noise).cpu()
            out_path = os.path.join(img_dir, f"samples_epoch_{epoch:03d}.png")
            vutils.save_image(denorm(fake_fixed), out_path, nrow=args.n_sample_grid, padding=2)
            print(f"[INFO] Saved sample grid to: {out_path}")

        # Save checkpoints
        if args.save_ckpt and (epoch % args.ckpt_every == 0 or epoch == args.epochs):
            torch.save({
                "epoch": epoch,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optG": optimizerG.state_dict(),
                "optD": optimizerD.state_dict(),
                "args": vars(args),
            }, os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:03d}.pt"))
            print(f"[INFO] Saved checkpoint @ epoch {epoch}")

        print(f"[EPOCH {epoch:03d}/{args.epochs}] G_loss={G_losses_epoch[-1]:.4f} | D_loss={D_losses_epoch[-1]:.4f}")

    # --------- Plot losses ---------
    fig = plt.figure(figsize=(8, 5))
    plt.plot(G_losses_iter, label='G loss (iter)')
    plt.plot(D_losses_iter, label='D loss (iter)', alpha=0.7)
    plt.title('GAN Training Loss (per iteration)')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    loss_iter_path = os.path.join(args.out_dir, 'loss_curve_iter.png')
    fig.savefig(loss_iter_path, dpi=150); plt.close(fig)
    print(f"[INFO] Saved iteration loss curve to: {loss_iter_path}")

    fig2 = plt.figure(figsize=(8, 5))
    plt.plot(G_losses_epoch, label='G loss (epoch)')
    plt.plot(D_losses_epoch, label='D loss (epoch)', alpha=0.7)
    plt.title('GAN Training Loss (per epoch)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    loss_epoch_path = os.path.join(args.out_dir, 'loss_curve_epoch.png')
    fig2.savefig(loss_epoch_path, dpi=150); plt.close(fig2)
    print(f"[INFO] Saved epoch loss curve to: {loss_epoch_path}")

    # --------- Optional GIF ---------
    if IMAGEIO_AVAILABLE and args.make_gif:
        images = []
        png_files = sorted([p for p in os.listdir(img_dir) if p.endswith('.png')])
        for fname in png_files:
            try:
                images.append(imageio.imread(os.path.join(img_dir, fname)))
            except Exception:
                pass
        gif_path = os.path.join(args.out_dir, 'training_progress.gif')
        if images:
            imageio.mimsave(gif_path, images, fps=min(6, max(1, len(images)//4)))
            print(f"[INFO] Saved GIF: {gif_path}")
        else:
            print("[WARN] No sample PNGs found to create GIF.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a basic GAN on MNIST and save losses & sample images.")
    parser.add_argument('--data-dir', type=str, default='./data', help='directory to download/load MNIST')
    parser.add_argument('--out-dir', type=str, default='./runs/mnist_gan', help='output directory for logs/images/ckpts')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for Adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--latent-dim', type=int, default=100, help='latent vector dimension')
    parser.add_argument('--ngf', type=int, default=64, help='generator feature maps')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator feature maps')
    parser.add_argument('--workers', type=int, default=2, help='dataloader workers')
    parser.add_argument('--n-sample-grid', type=int, default=8, help='grid size: produces n^2 samples')
    parser.add_argument('--sample-every', type=int, default=1, help='save generated samples every N epochs')
    parser.add_argument('--save-ckpt', action='store_true', help='save model checkpoints')
    parser.add_argument('--ckpt-every', type=int, default=5, help='checkpoint save interval in epochs')
    parser.add_argument('--make-gif', action='store_true', help='create a GIF from saved sample PNGs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--no-cuda', action='store_true', help='force CPU even if CUDA is available')
    parser.add_argument('--debug-shapes', action='store_true', help='print tensor shapes on first batch')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
