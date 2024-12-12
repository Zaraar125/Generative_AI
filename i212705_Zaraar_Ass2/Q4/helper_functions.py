import random, torch, os, numpy as np
import torch.nn as nn
import copy

LAMBDA_CYCLE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(model, optimizer, filename="models/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

def train_fn(
    disc_S, disc_P, gen_P, gen_S, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    S_reals = 0
    S_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (person, sketch) in enumerate(loop):
        person = person.to(DEVICE)
        sketch = sketch.to(DEVICE)

        # Train discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_sketch = gen_S(person)
            D_S_real = disc_S(sketch)
            D_S_fake = disc_S(fake_sketch.detach())
            S_reals += D_S_real.mean().item()
            S_fakes += D_S_fake.mean().item()
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            fake_person = gen_P(sketch)
            D_P_real = disc_P(person)
            D_P_fake = disc_P(fake_person.detach())
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            D_loss = (D_S_loss + D_P_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial losses
            D_S_fake = disc_S(fake_sketch)
            D_P_fake = disc_P(fake_person)
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))

            # cycle losses
            cycle_person = gen_P(fake_sketch)
            cycle_sketch = gen_S(fake_person)
            cycle_person_loss = l1(person, cycle_person)
            cycle_sketch_loss = l1(sketch, cycle_sketch)

            # identity losses
            # identity_zebra = gen_Z(zebra)
            # identity_horse = gen_H(horse)
            # identity_zebra_loss = l1(zebra, identity_zebra)
            # identity_horse_loss = l1(horse, identity_horse)

            # total loss
            G_loss = (
                loss_G_P
                + loss_G_S
                + cycle_person_loss * LAMBDA_CYCLE
                + cycle_sketch_loss * LAMBDA_CYCLE
                # + identity_horse_loss * LAMBDA_IDENTITY
                # + identity_zebra_loss * LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_sketch * 0.5 + 0.5, f"Training_Results/sketch_{idx}.png")
            save_image(fake_person * 0.5 + 0.5, f"Training_Results/person_{idx}.png")

        loop.set_postfix(S_real=S_reals / (idx + 1), S_fake=S_fakes / (idx + 1))

