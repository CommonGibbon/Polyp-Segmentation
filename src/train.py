# src/train.py
import subprocess
from pathlib import Path

import albumentations as A
import cv2
import hydra
import mlflow
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random

    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KvasirSegDataset(Dataset):
    """Kvasir-SEG dataset tells us how to load images and masks + apply augmentations."""

    def __init__(
        self,
        csv_path,
        size,
        train=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.df = pd.read_csv(csv_path)
        self.size = size
        self.train = train
        aug_train = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    translate_percent=0.05,  # ~ +/-5% shift in x and y
                    scale=(0.9, 1.1),  # 1 +/- 0.1
                    rotate=(-15, 15),  # +/- 15 degrees
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.ElasticTransform(
                    alpha=50,
                    sigma=7,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.2,
                ),
                A.Resize(size, size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
        aug_val = A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
        self.tf = aug_train if train else aug_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image"], cv2.IMREAD_COLOR)[:, :, ::-1]
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        out = self.tf(image=img, mask=mask)  # Apply transforms to the image and mask
        x = out["image"]
        y = out["mask"].unsqueeze(0)  # (1,H,W)
        return x, y


def dice_loss_logits(logits, targets, eps=1e-6):
    """This is the soft dice loss function we'll use for training."""
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def get_git_sha():
    """We'll log the git sha of the current commit as part of our MLflow experiment tracking."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return "unknown"


def overlay_and_save(x, y, yhat, path, thr=0.5):
    """Used for saving sample overlays images after training"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_np = (x.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    x_np = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)
    gt = (y.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    pr = (torch.sigmoid(yhat).squeeze(0).detach().cpu().numpy() > thr).astype(np.uint8)
    overlay = x_np.copy()
    overlay[gt == 1] = (0.5 * overlay[gt == 1] + 0.5 * np.array([0, 255, 0])).astype(
        np.uint8
    )
    overlay[pr == 1] = (0.5 * overlay[pr == 1] + 0.5 * np.array([0, 0, 255])).astype(
        np.uint8
    )
    cv2.imwrite(str(path), overlay)


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.train.seed)
    device = torch.device(
        cfg.train.device
        if torch.cuda.is_available() and cfg.train.device == "cuda"
        else "cpu"
    )

    train_ds = KvasirSegDataset(
        cfg.data.train_csv,
        cfg.data.size,
        train=True,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    val_ds = KvasirSegDataset(
        cfg.data.val_csv,
        cfg.data.size,
        train=False,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    model = smp.Unet(
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    ).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    dice_metric = BinaryF1Score(threshold=0.5).to(device)  # Dice == F1 for binary masks
    iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)

    mlflow.set_experiment(cfg.experiment.name)
    with mlflow.start_run():  # as run:
        mlflow.log_artifact(
            "dvc.lock"
        )  # track the dvc.lock as an experiment artifact so we can tell what data was used
        mlflow.log_params(
            {
                "encoder": cfg.model.encoder_name,
                "img_size": cfg.data.size,
                "batch_size": cfg.train.batch_size,
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
                "epochs": cfg.train.epochs,
                "amp": cfg.train.amp,
                "seed": cfg.train.seed,
            }
        )
        mlflow.set_tags(
            {
                "git_sha": get_git_sha(),
                "model_name": "unet",
                "dataset": "kvasir-seg",
            }
        )
        best_val = -1
        save_dir = Path(cfg.train.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        use_amp = device.type == "cuda" and cfg.train.amp
        for epoch in range(cfg.train.epochs):
            model.train()  # training mode - dropout is on, batchnorm uses batch stats, and ops are recorded for backprop
            tr_loss, tr_loss_bce, tr_loss_dice = 0.0, 0.0, 0.0
            for i, (x, y) in enumerate(train_dl):  # for each batch
                x, y = (
                    x.to(device, non_blocking=True),
                    y.to(device, non_blocking=True),
                )  # send to GPU (or CPU if GPU not available)
                opt.zero_grad(set_to_none=True)  # reset gradients between batches
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
                ):  # use lower precision for forward pass if using GPU and amp is enabled
                    logits = model(x)
                    bce_loss = bce(logits, y)
                    dice_loss = dice_loss_logits(logits, y)
                    loss = 0.5 * bce_loss + 0.5 * dice_loss
                loss.backward()
                # Optional: gradient clipping here if you use it
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                opt.step()

                tr_loss += loss.item()
                tr_loss_bce += bce_loss.item()
                tr_loss_dice += dice_loss.item()
                if (i + 1) % cfg.train.log_every_n_steps == 0:
                    step = epoch * len(train_dl) + i + 1
                    mlflow.log_metric(
                        "train_loss_total",
                        tr_loss / (i + 1),
                        step=step,
                    )
                    mlflow.log_metric(
                        "train_loss_bce",
                        tr_loss_bce / (i + 1),
                        step=step,
                    )
                    mlflow.log_metric(
                        "train_loss_dice",
                        tr_loss_dice / (i + 1),
                        step=step,
                    )

            # Validation
            model.eval()
            dice_metric.reset()
            iou_metric.reset()
            vl_loss = 0.0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = (
                        x.to(device, non_blocking=True),
                        y.to(device, non_blocking=True),
                    )
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
                    ):
                        logits = model(x)
                        loss = 0.5 * bce(logits, y) + 0.5 * dice_loss_logits(logits, y)
                        probs = torch.sigmoid(logits)
                    vl_loss += loss.item()
                    probs = probs.float()  # keep metrics in FP32 for stability
                    dice_metric.update(probs, y.int())
                    iou_metric.update(probs, y.int())
                val_dice = dice_metric.compute().item()
                val_iou = iou_metric.compute().item()
                mlflow.log_metrics(
                    {
                        "val_loss": vl_loss / len(val_dl),
                        "val_dice": val_dice,
                        "val_iou": val_iou,
                    },
                    step=epoch,
                )

            # Save sample overlays
            x0, y0 = next(iter(val_dl))
            x0, y0 = x0.to(device, non_blocking=True), y0.to(device, non_blocking=True)
            with torch.no_grad():
                logits0 = model(x0[:4])
            for k in range(min(4, x0.size(0))):
                overlay_and_save(
                    x0[k].cpu(),
                    y0[k].cpu(),
                    logits0[k].cpu(),
                    save_dir / f"overlay_e{epoch}_{k}.png",
                    thr=0.5,
                )
            if epoch == 0:
                for p in (
                    save_dir / f"overlay_e{epoch}_{k}.png"
                    for k in range(min(4, x0.size(0)))
                ):
                    mlflow.log_artifact(str(p))

            if val_dice > best_val:
                best_val = val_dice
                ckpt = save_dir / "best.pt"
                torch.save(
                    {"state_dict": model.state_dict(), "val_dice": val_dice}, ckpt
                )
                mlflow.log_artifact(str(ckpt))
                mlflow.log_metric("best_val_dice", best_val, step=epoch)
        print("Best val Dice:", best_val)


if __name__ == "__main__":
    main()
