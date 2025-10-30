# pix2pix.py
# ------------------------------------------------------------
# Pix2Pix training on Facades with:
#  - dataroot 자동 해결(.\facades\facades 같은 중첩 대응)
#  - (없으면) 자동 다운로드: datasets.ensure_facades 사용
#  - 손실 로그/그래프, 중간 샘플, 체크포인트 저장
# ------------------------------------------------------------
import os
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ★ 여기서 ensure_facades를 import해서 경로를 자동으로 맞춥니다.
from datasets import ImageDataset, ensure_facades
from models import GeneratorUNet, Discriminator, weights_init_normal


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def make_loaders(dataroot: str, image_size: int, batch_size: int, workers: int):
    tfms = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    train_set = ImageDataset(root=dataroot, transforms_=tfms, mode="train")
    test_set = ImageDataset(root=dataroot, transforms_=tfms, mode="test")

    if len(train_set) == 0 or len(test_set) == 0:
        raise RuntimeError(
            f"[data] empty dataset. train={len(train_set)}, test={len(test_set)}\n"
            f"  resolved dataroot = {dataroot}\n"
            f"  expected folders = {os.path.join(dataroot,'train')} / {os.path.join(dataroot,'test')}"
        )

    print(f"[data] #train: {len(train_set)}, #test: {len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def save_samples(G: nn.Module, batch: dict, device: torch.device, out_dir: str, tag: str):
    G.eval()
    with torch.no_grad():
        A = batch["A"].to(device)
        B = batch["B"].to(device)
        fakeB = G(A)
        grid = torch.cat([A, fakeB, B], dim=0)
        grid = denorm(grid)
        grid = make_grid(grid, nrow=A.size(0))
        os.makedirs(out_dir, exist_ok=True)
        save_image(grid, os.path.join(out_dir, f"samples_{tag}.png"))
    G.train()


def plot_losses(g_hist: List[float], d_hist: List[float], l1_hist: List[float], out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(g_hist, label="G_total")
    plt.plot(l1_hist, label="L1")
    plt.plot(d_hist, label="D")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[pix2pix] device = {device}")

    # ★ dataroot 자동 해결(+필요 시 다운로드까지)
    resolved_root = ensure_facades(args.dataroot)
    print(f"[pix2pix] resolved dataroot: {resolved_root}")

    # loaders
    train_loader, test_loader = make_loaders(
        dataroot=resolved_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    fixed_val = next(iter(test_loader))

    # models
    netG = GeneratorUNet(in_channels=3, out_channels=3).to(device)
    netD = Discriminator(in_channels=3).to(device)
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    # losses & opt
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # outs
    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images")
    ckpt_dir = os.path.join(args.out_dir, "saved_models")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log_pix2pix.txt")

    g_hist, d_hist, l1_hist = [], [], []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for _, batch in enumerate(train_loader, start=1):
            A = batch["A"].to(device, non_blocking=True)
            B = batch["B"].to(device, non_blocking=True)

            # ---- D ----
            optD.zero_grad()
            pred_real = netD(A, B)
            loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real, device=device))
            with torch.no_grad():
                fake_B = netG(A)
            pred_fake = netD(A, fake_B.detach())
            loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake, device=device))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optD.step()

            # ---- G ----
            optG.zero_grad()
            fake_B = netG(A)
            pred_fake_for_G = netD(A, fake_B)
            loss_G_GAN = criterion_gan(pred_fake_for_G, torch.ones_like(pred_fake_for_G, device=device))
            loss_G_L1 = criterion_l1(fake_B, B) * args.lambda_l1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optG.step()

            # logs
            global_step += 1
            g_hist.append(loss_G.item())
            d_hist.append(loss_D.item())
            l1_hist.append((loss_G_L1 / args.lambda_l1).item())

            if global_step % args.log_every == 0:
                msg = (f"[Epoch {epoch}/{args.epochs}] "
                       f"Iter {global_step:06d} | "
                       f"D: {loss_D.item():.4f} | "
                       f"G: {loss_G.item():.4f} (GAN {loss_G_GAN.item():.4f} + "
                       f"L1 {(loss_G_L1.item()/args.lambda_l1):.4f})")
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

            if global_step % args.sample_every == 0:
                save_samples(netG, fixed_val, device, img_dir, f"e{epoch}_it{global_step}")

        # epoch end
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save(netG.state_dict(), os.path.join(ckpt_dir, f"netG_e{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(ckpt_dir, f"netD_e{epoch}.pth"))

        plot_losses(g_hist, d_hist, l1_hist, os.path.join(args.out_dir, "loss_curve.png"))

    print("[pix2pix] done.")
    print(f" samples -> {img_dir}")
    print(f" ckpts   -> {ckpt_dir}")
    print(f" curve   -> {os.path.join(args.out_dir, 'loss_curve.png')}")
    print(f" log     -> {log_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, default="./facades",
                   help="dataset root (train/test가 들어있는 폴더 또는 그 상위)")
    p.add_argument("--out_dir", type=str, default="./", help="output dir")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
