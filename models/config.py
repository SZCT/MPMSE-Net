import json
from dataclasses import dataclass, field
from pathlib import Path


def _to_path(value):
    return value if isinstance(value, Path) else Path(value)


def _to_tuple(value, default):
    if value is None:
        return default
    return tuple(value)


@dataclass
class DataConfig:
    split_dir: Path = Path("Split_info")
    model_meta: Path = Path("japan_model_meta.json")
    train_noise: bool = True
    valid_noise: bool = True
    wavetype: str = "both"
    time_length: int = 180
    n_stations: int = 50
    min_removed_stations: int = 10
    min_station_dist: tuple[int, int, int] = (6, 3, 6)
    final_size: tuple[int, int] = (30, 40)
    slip_area_classes: int = 2
    shuffle_stations: bool = True

    @classmethod
    def from_dict(cls, values):
        payload = dict(values)
        if "split_dir" in payload:
            payload["split_dir"] = _to_path(payload["split_dir"])
        if "model_meta" in payload:
            payload["model_meta"] = _to_path(payload["model_meta"])
        payload["min_station_dist"] = _to_tuple(payload.get("min_station_dist"), (6, 3, 6))
        payload["final_size"] = _to_tuple(payload.get("final_size"), (30, 40))
        return cls(**payload)


@dataclass
class ModelConfig:
    task_mode: str = "all"
    final_size: tuple[int, int] = (30, 40)
    freeze_magnitude: bool = False
    freeze_max_slip: bool = False
    freeze_slip: bool = False

    @classmethod
    def from_dict(cls, values):
        payload = dict(values)
        payload["final_size"] = _to_tuple(payload.get("final_size"), (30, 40))
        return cls(**payload)


@dataclass
class TrainConfig:
    output_dir: Path = Path("runs/example")
    log_dir: Path = Path("logs/example")
    epochs: int = 200
    batch_size: int = 64
    seed: int = 518
    patience: int = 20
    num_workers: int = 0
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {
            "m": 1e-4,
            "max_slip": 1e-4,
            "slip": 1e-3,
            "all": 1e-4,
        }
    )
    weight_decay: float = 0.0
    area_weight: float = 1.0
    slip_weight: float = 10.0
    slip_ssim_weight: float = 1.0
    max_slip_weight: float = 1.0
    final_slip_weight: float = 1.0
    final_ssim_weight: float = 1.0
    average_slip_weight: float = 10.0
    resume_checkpoint: Path | None = None

    @classmethod
    def from_dict(cls, values):
        payload = dict(values)
        if "output_dir" in payload:
            payload["output_dir"] = _to_path(payload["output_dir"])
        if "log_dir" in payload:
            payload["log_dir"] = _to_path(payload["log_dir"])
        if payload.get("resume_checkpoint") is not None:
            payload["resume_checkpoint"] = _to_path(payload["resume_checkpoint"])
        return cls(**payload)


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_json(cls, path):
        path = _to_path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            data=DataConfig.from_dict(payload.get("data", {})),
            model=ModelConfig.from_dict(payload.get("model", {})),
            train=TrainConfig.from_dict(payload.get("train", {})),
        )
