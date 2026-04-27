import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import DataConfig, ModelConfig, TrainConfig
from .data import build_dataset, load_model_meta
from .losses import ssim_loss, weighted_average_slip_loss
from .metrics import scalar_metrics, slip_area_metrics, slip_distribution_metrics
from .model import MultiTaskSlipModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mu_area(path, dip_dim=30, strike_dim=40, device="cpu"):
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    expected = dip_dim * strike_dim
    if data.shape[0] != expected:
        raise ValueError(f"Expected {expected} rupture patches, got {data.shape[0]}")
    area = data[:, 10] * data[:, 11]
    mu = data[:, 13]
    mu = torch.tensor(mu.reshape(dip_dim, strike_dim), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    area = torch.tensor(area.reshape(dip_dim, strike_dim), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return {"mu": mu, "area": area, "mu_area_sum": (mu * area).sum()}


def extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)


class Trainer:
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, train_config: TrainConfig):
        set_seed(train_config.seed)
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_meta = load_model_meta(data_config.model_meta)
        self.model = MultiTaskSlipModel(model_config).to(self.device)
        self.load_resume_checkpoint()

        self.optimizer = self.build_optimizer(self.learning_rate_for_epoch(0))
        self.current_lr = self.learning_rate_for_epoch(0)

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        self.train_config.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_config.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.train_config.log_dir))

        self.train_loader, self.valid_loader = self.build_loaders()
        self.mu_area_cache = [load_mu_area(item["rupt_info"], device="cpu") for item in self.model_meta]
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def build_optimizer(self, learning_rate):
        return AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.train_config.weight_decay,
        )

    def learning_rate_for_epoch(self, epoch):
        mode = self.model_config.task_mode
        base_lr = self.train_config.learning_rates[mode]
        if mode == "slip":
            return 1e-4 if epoch >= 60 else base_lr
        if mode == "all":
            return 1e-5 if epoch >= 60 else base_lr
        if mode in {"m", "max_slip"}:
            if epoch >= 200:
                return 1e-6
            if epoch >= 60:
                return 1e-5
        return base_lr

    def maybe_update_optimizer(self, epoch):
        new_lr = self.learning_rate_for_epoch(epoch)
        if np.isclose(new_lr, self.current_lr):
            return
        self.optimizer = self.build_optimizer(new_lr)
        self.current_lr = new_lr

    def build_loaders(self):
        train_dataset = build_dataset("train", self.data_config, self.model_meta)
        valid_dataset = build_dataset("valid", self.data_config, self.model_meta)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, valid_loader

    def load_resume_checkpoint(self):
        checkpoint_path = self.train_config.resume_checkpoint
        if checkpoint_path is None:
            return
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(extract_state_dict(payload), strict=False)

    def to_device(self, batch):
        return tuple(item.to(self.device, non_blocking=True) if torch.is_tensor(item) else item for item in batch)

    def forward_batch(self, batch):
        x_disp, max_disp, x_vel, max_vel, loc_disp, loc_vel, y_mw, y_zone, y_slip, y_slip_max, y_slip_final, model_id = batch
        y_slip = y_slip.unsqueeze(1)
        y_slip_final = y_slip_final.unsqueeze(1)
        outputs = self.model(x_disp, x_vel, y_mw, max_disp, max_vel, loc_disp, loc_vel)
        return outputs, {
            "mw": y_mw,
            "zone": y_zone.long(),
            "slip": y_slip,
            "slip_max": y_slip_max.view(-1),
            "slip_final": y_slip_final,
            "model_id": model_id,
        }

    def loss_terms(self, outputs, target):
        cfg = self.train_config
        losses = {}
        if self.model_config.task_mode == "m":
            losses["magnitude"] = self.mae(outputs["magnitude"], target["mw"])
        if "slip_area" in outputs:
            losses["area"] = cfg.area_weight * self.ce(outputs["slip_area"], target["zone"])
        if "slip_norm" in outputs:
            slip_mse = self.mse(outputs["slip_norm"], target["slip"])
            slip_ssim = ssim_loss(outputs["slip_norm"], target["slip"])
            losses["slip"] = cfg.slip_weight * slip_mse + cfg.slip_ssim_weight * slip_ssim
        if "max_slip" in outputs:
            losses["max_slip"] = cfg.max_slip_weight * self.mse(outputs["max_slip"].squeeze(-1), target["slip_max"])
        if self.model_config.task_mode == "all":
            final_mse = self.mse(outputs["slip_final"], target["slip_final"])
            final_ssim = ssim_loss(outputs["slip_final"], target["slip_final"])
            avg_slip = weighted_average_slip_loss(outputs["slip_final"], target["slip_final"], target["model_id"], self.mu_area_cache)
            losses["final_slip"] = cfg.final_slip_weight * final_mse + cfg.final_ssim_weight * final_ssim
            losses["average_slip"] = cfg.average_slip_weight * avg_slip
        losses["total"] = sum(losses.values())
        return losses

    def batch_metrics(self, outputs, target):
        rows = {}
        if "magnitude" in outputs:
            rows.update({f"magnitude_{key}": value for key, value in scalar_metrics(outputs["magnitude"], target["mw"]).items()})
        if "slip_area" in outputs:
            rows.update(slip_area_metrics(outputs["slip_area"], target["zone"]))
        if "slip_norm" in outputs:
            mask = (target["slip"] > 0.02).float()
            rows.update(slip_distribution_metrics(outputs["slip_norm"], target["slip"], mask))
        if "max_slip" in outputs:
            pred = torch.exp(outputs["max_slip"].squeeze(-1)) - 1
            true = torch.exp(target["slip_max"]) - 1
            rows.update({f"max_slip_{key}": value for key, value in scalar_metrics(pred, true).items()})
        if "slip_final" in outputs:
            mask = (target["slip_final"] > 0.02).float()
            rows.update({f"final_{key}": value for key, value in slip_distribution_metrics(outputs["slip_final"], target["slip_final"], mask).items()})
        return rows

    def merge_metric_rows(self, rows, prefix):
        merged = {}
        keys = sorted({key for row in rows for key in row})
        for key in keys:
            merged[f"{prefix}/{key}"] = nanmean([row[key] for row in rows if key in row])
        return merged

    def write_metrics(self, metrics, epoch):
        for key, value in metrics.items():
            if np.isfinite(value):
                self.writer.add_scalar(key, value, epoch)

    def run_epoch(self, loader, training):
        self.model.train(mode=training)
        loss_meters = {}
        metric_rows = []

        for batch in loader:
            batch = self.to_device(batch)
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                outputs, target = self.forward_batch(batch)
                losses = self.loss_terms(outputs, target)
                if training:
                    losses["total"].backward()
                    self.optimizer.step()

            batch_size = target["mw"].size(0)
            for key, value in losses.items():
                loss_meters.setdefault(key, AverageMeter()).update(value.detach().item(), batch_size)
            if not training:
                metric_rows.append(self.batch_metrics(outputs, target))

        phase = "train" if training else "valid"
        metrics = {f"{phase}/{key}": meter.avg for key, meter in loss_meters.items()}
        if not training:
            metrics.update(self.merge_metric_rows(metric_rows, phase))
        return metrics

    def save_checkpoint(self, epoch, metrics):
        model_path = self.train_config.output_dir / f"best_model_{self.model_config.task_mode}.pth"
        metrics_path = self.train_config.output_dir / f"best_metrics_{self.model_config.task_mode}.json"
        torch.save(self.model.state_dict(), model_path)
        payload = {"epoch": epoch + 1, **metrics}
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def fit(self):
        for epoch in range(self.train_config.epochs):
            self.maybe_update_optimizer(epoch)
            train_metrics = self.run_epoch(self.train_loader, training=True)
            valid_metrics = self.run_epoch(self.valid_loader, training=False)

            self.write_metrics(train_metrics, epoch)
            self.write_metrics(valid_metrics, epoch)

            current = valid_metrics["valid/total"]
            print(
                f"Epoch {epoch + 1}/{self.train_config.epochs} "
                f"lr={self.current_lr:.1e} "
                f"train={train_metrics['train/total']:.6f} "
                f"valid={current:.6f}"
            )

            if current < self.best_loss:
                self.best_loss = current
                self.bad_epochs = 0
                self.save_checkpoint(epoch, valid_metrics)
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.train_config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.writer.close()
