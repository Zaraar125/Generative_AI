import torch
from Cycle_GAN import Generator,Discriminator
from torch.optim import optim
import torch.nn as nn
from helper_functions import load_checkpoint,save_checkpoint,train_fn
from Custom_Dataset import PersonSketchDataset
from Transformations import transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

TRAIN_DIR = "/kaggle/input/person-face-sketches/train"
VAL_DIR = "/kaggle/input/person-face-sketches/val"

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
NUM_WORKERS = 4
NUM_EPOCHS = 3

LOAD_MODEL = False

SAVE_MODEL = True

CHECKPOINT_GENERATOR_S = "models/genS.pth.tar"
CHECKPOINT_GENERATOR_P = "models/genP.pth.tar"
CHECKPOINT_DISCRIMINATOR_S = "models/discS.pth.tar"
CHECKPOINT_DISCRIMINATOR_P = "models/discP.pth.tar"


disc_S = Discriminator(in_channels=3).to(DEVICE)
disc_P = Discriminator(in_channels=3).to(DEVICE)
gen_P = Generator(img_channels=3, num_residuals=9).to(DEVICE)
gen_S = Generator(img_channels=3, num_residuals=9).to(DEVICE)

# use Adam Optimizer for both generator and discriminator
opt_disc = optim.Adam(
    list(disc_S.parameters()) + list(disc_P.parameters()),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)

opt_gen = optim.Adam(
    list(gen_P.parameters()) + list(gen_S.parameters()),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)

L1 = nn.L1Loss()
mse = nn.MSELoss()

if LOAD_MODEL:
    load_checkpoint(CHECKPOINT_GENERATOR_S,gen_S,opt_gen,LEARNING_RATE,)
    load_checkpoint( CHECKPOINT_GENERATOR_P,gen_P,opt_gen, LEARNING_RATE,)
    load_checkpoint(CHECKPOINT_DISCRIMINATOR_S,disc_S,opt_disc,LEARNING_RATE,)
    load_checkpoint(CHECKPOINT_DISCRIMINATOR_P,disc_P,opt_disc,LEARNING_RATE,)

dataset = PersonSketchDataset(
    root_person=TRAIN_DIR + "/photos",
    root_sketch=TRAIN_DIR + "/sketches",
    transform=transforms,
)
val_dataset = PersonSketchDataset(
    root_person=VAL_DIR + "/photos",
    root_sketch=VAL_DIR + "/sketches",
    transform=transforms,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

for epoch in range(NUM_EPOCHS):
    train_fn(disc_S,disc_P,gen_P,gen_S,loader,opt_disc,opt_gen,L1,mse,d_scaler,g_scaler,)

    if SAVE_MODEL:
        save_checkpoint(gen_S, opt_gen, filename=CHECKPOINT_GENERATOR_S)
        save_checkpoint(gen_P, opt_gen, filename=CHECKPOINT_GENERATOR_P)
        save_checkpoint(disc_S, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_S)
        save_checkpoint(disc_P, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_P)
