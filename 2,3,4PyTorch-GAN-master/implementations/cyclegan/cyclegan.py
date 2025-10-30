# cyclegan.py (Windows-safe full version)
import argparse
import os
import sys
import numpy as np
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_opts():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    p.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    p.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
    p.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    p.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    p.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    p.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    p.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    # ★ Windows 안전: 기본 0
    p.add_argument("--n_cpu", type=int, default=0, help="number of cpu workers for dataloader (Windows=0 권장)")
    p.add_argument("--img_height", type=int, default=256, help="size of image height")
    p.add_argument("--img_width", type=int, default=256, help="size of image width")
    p.add_argument("--channels", type=int, default=3, help="number of image channels")
    p.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    p.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
    p.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    p.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    p.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    return p.parse_args()


def main():
    opt = parse_opts()
    print(opt)

    # Create sample and checkpoint directories
    os.makedirs(f"images/{opt.dataset_name}", exist_ok=True)
    os.makedirs(f"saved_models/{opt.dataset_name}", exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)

    if cuda:
        criterion_GAN = criterion_GAN.cuda()
        criterion_cycle = criterion_cycle.cuda()
        criterion_identity = criterion_identity.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/G_AB_{opt.epoch}.pth", map_location=device))
        G_BA.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/G_BA_{opt.epoch}.pth", map_location=device))
        D_A.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/D_A_{opt.epoch}.pth", map_location=device))
        D_B.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/D_B_{opt.epoch}.pth", map_location=device))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Image transformations
    transforms_ = [
        transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # -------------------------------
    #  Dataset Loaders (data/%s)
    # -------------------------------
    train_root = os.path.join("data", opt.dataset_name)
    print(f"[debug] looking for dataset at: {train_root}")
    for sub in ["trainA", "trainB", "testA", "testB"]:
        d = os.path.join(train_root, sub)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f" {sub:<6}: {n}")

    dataloader = DataLoader(
        ImageDataset(f"data/{opt.dataset_name}", transforms_=transforms_, unaligned=True, mode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,      # ★ Windows는 0 권장
        pin_memory=cuda,
    )
    val_dataloader = DataLoader(
        ImageDataset(f"data/{opt.dataset_name}", transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=5,
        shuffle=True,
        num_workers=0,
        pin_memory=cuda,
    )

    def sample_images(batches_done):
        """Saves a generated sample from the test set"""
        imgs = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A_g = make_grid(real_A, nrow=5, normalize=True)
        real_B_g = make_grid(real_B, nrow=5, normalize=True)
        fake_A_g = make_grid(fake_A, nrow=5, normalize=True)
        fake_B_g = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A_g, fake_B_g, real_B_g, fake_A_g), 1)
        save_image(image_grid, f"images/{opt.dataset_name}/{batches_done}.png", normalize=False)
        G_AB.train()
        G_BA.train()

    # ----------
    #  Training
    # ----------
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------
            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f, adv: %.4f, cycle: %.4f, identity: %.4f] ETA: %s"
                % (
                    epoch, opt.n_epochs, i, len(dataloader),
                    loss_D.item(), loss_G.item(), loss_GAN.item(), loss_cycle.item(), loss_identity.item(),
                    time_left,
                )
            )
            sys.stdout.flush()

            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(G_AB.state_dict(), f"saved_models/{opt.dataset_name}/G_AB_{epoch}.pth")
            torch.save(G_BA.state_dict(), f"saved_models/{opt.dataset_name}/G_BA_{epoch}.pth")
            torch.save(D_A.state_dict(), f"saved_models/{opt.dataset_name}/D_A_{epoch}.pth")
            torch.save(D_B.state_dict(), f"saved_models/{opt.dataset_name}/D_B_{epoch}.pth")


if __name__ == "__main__":
    # Windows 멀티프로세싱 안전 가드
    # (num_workers>0를 쓰더라도 main 가드가 있어야 함)
    main()
