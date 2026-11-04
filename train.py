import argparse
import random
import numpy as np
import torch
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from dataloader.gafa import create_gafa_dataset
from models.hbonet_1 import HBONet
from models.gazenet_1 import GazeNet



torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def train(opt):
    seed_everything(42, workers=True)

    # ---------- 1. Create HBNet ----------
    hbonet = HBONet()


    # ---------- 2. Create GazeNet ----------
    model = GazeNet( hbonet, learning_rate=1e-4)

    # ---------- 3. Dataset ----------
    train_exp_names = [
        'library/1026_3', 'library/1028_2', 'library/1028_5',
        'lab/1013_1', 'lab/1014_1', 'kitchen/1022_4', 'kitchen/1015_4',
        'living_room/004', 'living_room/005', 'courtyard/004', 'courtyard/005',
    ]
    random.shuffle(train_exp_names)   

    dset = create_gafa_dataset(
        n_frames=opt.n_frames,
        exp_names=train_exp_names,
        interval=1,
    )

    n_total = len(dset)
    train_idx = np.arange(0, int(n_total * 0.9))
    val_idx = np.arange(int(n_total * 0.9), n_total)
    train_dset = Subset(dset, train_idx)
    val_dset = Subset(dset, val_idx)

    print("=" * 60)
    print(f"Total samples : {n_total}")
    print(f"Train samples : {len(train_dset)}")
    print(f"Val samples   : {len(val_dset)}")
    print("=" * 60)

    # ---------- 4. Callbacks ----------
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        save_top_k=3,   
        filename="epoch={epoch}-val_mae={val_mae:.4f}",
        auto_insert_metric_name=False
    )

    # ---------- 5. Trainer ----------
    trainer = Trainer(
        default_root_dir=opt.checkpoint,
        callbacks=[checkpoint_callback],
        max_epochs=opt.epoch,
        accelerator='gpu',
        devices=1,
        precision=16,
        benchmark=False,      
        deterministic=False,  
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ---------- 6. Data Loading ----------
    train_loader = DataLoader(
        train_dset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---------- 7. Start Training ----------
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default="output/ablation")
    opt = parser.parse_args()
    train(opt)
