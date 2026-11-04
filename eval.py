import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from dataloader.gafa import create_gafa_dataset
from models.hbonet_1  import HBONet
from models.gazenet_1 import GazeNet

test_exp_names = [
    'library/1029_2',
    'lab/1013_2',
    'kitchen/1022_2',
    'living_room/006',
    'courtyard/002',
    'courtyard/003',
]


def test_single_checkpoint(ckpt_path, trainer, test_loader):
    print(f"\n{'='*80}")
    print(f"Testing model: {os.path.basename(ckpt_path)}")
    print("=" * 80)

    # Manually create the model 
    hbonet = HBONet()
    model = GazeNet(
        hbonet,
        learning_rate=5e-5
    )

    # Load weights
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(" Weights loaded successfully!")

    # Test
    trainer.test(model, dataloaders=test_loader, verbose=False)

    results = model.final_test_results
    mae_3d_all = results["MAE (3D all)"]
    mae_2d_all = results["MAE (2D all)"]

    print(f" Test results:")
    print(f"MAE (3D all): {mae_3d_all:.4f}°")
    print(f"MAE (2D all): {mae_2d_all:.4f}°")

    return mae_3d_all, os.path.basename(ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory containing .ckpt files")
    opt = parser.parse_args()

    # Test dataset
    print("Building official test dataset...")
    test_dset = create_gafa_dataset(
        n_frames=opt.n_frames,
        exp_names=test_exp_names,
        interval=7,
    )
    test_loader = DataLoader(test_dset, batch_size=16, num_workers=4, shuffle=False)
    print(f" Test dataset built successfully, number of samples: {len(test_dset)}")

    # Trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=opt.gpus,
        precision=16,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Iterate through checkpoints
    ckpt_dir = opt.ckpt_dir
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    ckpt_files = [
        os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and "last" not in f
    ]
    ckpt_files.sort()
    

    all_results = []
    for ckpt_path in ckpt_files:
        mae, ckpt_name = test_single_checkpoint(ckpt_path, trainer, test_loader)
        all_results.append((ckpt_name, mae))

    # Summary
    print("\n" + "=" * 80)
    print(" Summary of all model test results (sorted by 3D MAE ascending)")
    print("=" * 80)
    all_results.sort(key=lambda x: x[1])
    for ckpt_name, mae in all_results:
        print(f"{ckpt_name:40s} | MAE (3D all): {mae:.4f}°")
    print("=" * 80)
    if all_results:
        print(f"🏆 Best model: {all_results[0][0]}, MAE: {all_results[0][1]:.4f}°")


if __name__ == "__main__":
    main()
