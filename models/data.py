import json
from pathlib import Path

import numpy as np
import obspy
import torch
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize
from torch.utils.data import Dataset


def as_path(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode()
    return str(value)


def resolve_path(value, base_dir=None):
    path = Path(as_path(value))
    if path.is_absolute() or base_dir is None:
        return path
    return (Path(base_dir) / path).resolve()


def load_station_file(path):
    data = np.genfromtxt(path, "S12", skip_header=1)
    return {row[0].decode(): [float(row[1]), float(row[2])] for row in data}


def normalize_by_station_amplitude(x):
    max_amp = np.max(np.abs(x), axis=(0, 1))
    out = np.zeros_like(x, dtype=np.float32)
    mask = max_amp > 0
    out[:, :, mask] = x[:, :, mask] / max_amp[mask][None, None, :]
    return out, max_amp.astype(np.float32)


def stack_components(e, n, z):
    return np.transpose(np.dstack((e, n, z)), (2, 0, 1)).astype(np.float32)


def segment_slip(y, num_classes=2):
    y = torch.as_tensor(y, dtype=torch.float32)
    max_value = torch.max(y)
    labels = torch.zeros_like(y, dtype=torch.long)
    if num_classes == 2:
        labels[y > 0.02 * max_value] = 1
    else:
        labels[(y > 0.05 * max_value) & (y <= 0.25 * max_value)] = 1
        labels[(y > 0.25 * max_value) & (y <= 0.50 * max_value)] = 2
        labels[(y > 0.50 * max_value) & (y <= 0.75 * max_value)] = 3
        labels[y > 0.75 * max_value] = 4
    return labels


def normalize_slip(y, eps=1e-8):
    return y / (np.max(y) + eps)


def resize_slip(y, output_size):
    y = y.copy()
    ys, xs = np.nonzero(y > 0)
    if len(ys) == 0:
        empty = np.zeros(output_size, dtype=np.float32)
        return y.astype(np.float32), empty, empty
    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    crop = y[top : bottom + 1, left : right + 1]
    mask = crop > 0
    _, indices = distance_transform_edt(~mask, return_indices=True)
    filled_crop = crop[indices[0], indices[1]]
    full = y.copy()
    full[top : bottom + 1, left : right + 1] = filled_crop
    resized = resize(filled_crop, output_size, order=3, mode="edge", anti_aliasing=True).astype(np.float32)
    return full.astype(np.float32), resized, normalize_slip(resized).astype(np.float32)


class EarthquakeDataset(Dataset):
    def __init__(
        self,
        e_paths,
        n_paths,
        z_paths,
        y_path,
        eqinfo,
        station_disp_files,
        station_vel_files,
        loc_disp_paths,
        loc_vel_paths,
        time_length=180,
        n_stations=50,
        min_removed_stations=10,
        min_station_dist=(6, 3, 6),
        wavetype="both",
        final_size=(30, 40),
        slip_area_classes=2,
        shuffle_stations=True,
    ):
        self.e_paths = e_paths
        self.n_paths = n_paths
        self.z_paths = z_paths
        self.loc_disp_paths = loc_disp_paths
        self.loc_vel_paths = loc_vel_paths
        self.y_data = np.load(y_path, allow_pickle=True).item()
        self.eqinfo = eqinfo
        self.station_disp = [load_station_file(path) for path in station_disp_files]
        self.station_vel = [load_station_file(path) for path in station_vel_files]
        self.time_length = time_length
        self.n_stations = n_stations
        self.min_removed_stations = min_removed_stations
        self.min_station_dist = min_station_dist
        self.wavetype = wavetype
        self.final_size = final_size
        self.slip_area_classes = slip_area_classes
        self.shuffle_stations = shuffle_stations

    def __len__(self):
        return len(self.e_paths)

    def load_wave(self, path):
        return np.load(as_path(path)).T[: self.time_length].astype(np.float32)

    def load_loc(self, path):
        loc = np.load(as_path(path)).astype(np.float32)
        if loc.shape != (self.n_stations, 4):
            raise ValueError(f"Expected station location shape {(self.n_stations, 4)}, got {loc.shape}")
        return loc

    def station_distances(self, station_dict, eqlon, eqlat, eqdep):
        distances = []
        for lon, lat in station_dict.values():
            epi = obspy.geodetics.locations2degrees(lat1=eqlat, long1=eqlon, lat2=lat, long2=lon)
            dep = abs(eqdep) / (6371 * np.pi / 180)
            distances.append(np.sqrt(epi**2 + dep**2))
        return np.asarray(distances)

    def removed_station_indices(self, distances):
        selectable = np.where(distances < self.min_station_dist[2])[0]
        if len(selectable) < self.min_removed_stations:
            selectable = np.arange(self.n_stations)
        while True:
            n_remove = np.random.randint(self.min_removed_stations, len(selectable) + 1)
            remove_idx = np.random.choice(selectable, n_remove, replace=False)
            if len(np.where(distances[remove_idx] < self.min_station_dist[1])[0]) >= self.min_station_dist[0]:
                return remove_idx

    def load_streams(self, index):
        if self.wavetype == "disp":
            e_disp = self.load_wave(self.e_paths[index])
            n_disp = self.load_wave(self.n_paths[index])
            z_disp = self.load_wave(self.z_paths[index])
            e_vel = n_vel = z_vel = np.zeros((self.time_length, self.n_stations), dtype=np.float32)
        elif self.wavetype == "vel":
            e_vel = self.load_wave(self.e_paths[index])
            n_vel = self.load_wave(self.n_paths[index])
            z_vel = self.load_wave(self.z_paths[index])
            e_disp = n_disp = z_disp = np.zeros((self.time_length, self.n_stations), dtype=np.float32)
        elif self.wavetype == "both":
            e_disp, e_vel = self.e_paths[index]
            n_disp, n_vel = self.n_paths[index]
            z_disp, z_vel = self.z_paths[index]
            e_disp = self.load_wave(e_disp)
            n_disp = self.load_wave(n_disp)
            z_disp = self.load_wave(z_disp)
            e_vel = self.load_wave(e_vel)
            n_vel = self.load_wave(n_vel)
            z_vel = self.load_wave(z_vel)
        else:
            raise ValueError("wavetype must be one of: disp, vel, both")
        return e_disp, n_disp, z_disp, e_vel, n_vel, z_vel

    def process_stream(self, e, n, z, loc, station_dict, eqlon, eqlat, eqdep):
        distances = self.station_distances(station_dict, eqlon, eqlat, eqdep)
        remove_idx = self.removed_station_indices(distances)
        for arr in (e, n, z):
            arr[:, remove_idx] = 0.0
        x, max_amp = normalize_by_station_amplitude(stack_components(e, n, z))
        if self.shuffle_stations:
            perm = np.random.permutation(self.n_stations)
            x = x[:, :, perm]
            loc = loc[perm]
            max_amp = max_amp[perm]
        return x, max_amp, loc

    def __getitem__(self, index):
        row = self.eqinfo[index]
        eqlon, eqlat, eqdep = float(row[2]), float(row[3]), float(row[4])
        model_id = int(row[-1]) - 1
        e_disp, n_disp, z_disp, e_vel, n_vel, z_vel = self.load_streams(index)
        loc_disp = self.load_loc(self.loc_disp_paths[index]) if self.wavetype in {"disp", "both"} else np.zeros((self.n_stations, 4), dtype=np.float32)
        loc_vel = self.load_loc(self.loc_vel_paths[index]) if self.wavetype in {"vel", "both"} else np.zeros((self.n_stations, 4), dtype=np.float32)
        if self.wavetype in {"disp", "both"}:
            x_disp, max_disp, loc_disp = self.process_stream(e_disp, n_disp, z_disp, loc_disp, self.station_disp[model_id], eqlon, eqlat, eqdep)
        else:
            x_disp = np.zeros((3, self.time_length, self.n_stations), dtype=np.float32)
            max_disp = np.ones((self.n_stations,), dtype=np.float32)
        if self.wavetype in {"vel", "both"}:
            x_vel, max_vel, loc_vel = self.process_stream(e_vel, n_vel, z_vel, loc_vel, self.station_vel[model_id], eqlon, eqlat, eqdep)
        else:
            x_vel = np.zeros((3, self.time_length, self.n_stations), dtype=np.float32)
            max_vel = np.ones((self.n_stations,), dtype=np.float32)
        slip = self.y_data["slip"][index]
        slip_max = np.float32(np.log(np.max(slip) + 1))
        slip_full, slip_resized, slip_norm = resize_slip(slip, self.final_size)
        zone = segment_slip(slip_full, self.slip_area_classes)
        return (
            torch.tensor(x_disp, dtype=torch.float32),
            torch.tensor(max_disp, dtype=torch.float32),
            torch.tensor(x_vel, dtype=torch.float32),
            torch.tensor(max_vel, dtype=torch.float32),
            torch.tensor(loc_disp, dtype=torch.float32),
            torch.tensor(loc_vel, dtype=torch.float32),
            torch.tensor(self.y_data["Mw"][index], dtype=torch.float32),
            zone,
            torch.tensor(slip_norm, dtype=torch.float32),
            torch.tensor(slip_max, dtype=torch.float32),
            torch.tensor(slip_resized, dtype=torch.float32),
            torch.tensor(model_id, dtype=torch.long),
        )


def load_model_meta(path):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        model_meta = json.load(f)
    base_dir = path.parent
    for item in model_meta:
        for key in ("gnss_loc", "sm_loc", "rupt_info", "fault_loc"):
            if key in item and item[key] is not None:
                item[key] = str(resolve_path(item[key], base_dir))
    return model_meta


def split_file(split_dir, split, name, use_noise=False):
    suffix = "_noise" if use_noise and name in {"E", "N", "Z"} else ""
    return Path(split_dir) / f"X_{split}_{name}{suffix}.npy"


def build_dataset(split, data_config, model_meta):
    split_dir = Path(data_config.split_dir)
    use_noise = data_config.train_noise if split == "train" else data_config.valid_noise
    station_disp_files = [m["gnss_loc"] for m in model_meta]
    station_vel_files = [m["sm_loc"] for m in model_meta]
    return EarthquakeDataset(
        e_paths=np.load(split_file(split_dir, split, "E", use_noise), allow_pickle=True),
        n_paths=np.load(split_file(split_dir, split, "N", use_noise), allow_pickle=True),
        z_paths=np.load(split_file(split_dir, split, "Z", use_noise), allow_pickle=True),
        y_path=split_dir / f"y_{split}.npy",
        eqinfo=np.load(split_dir / f"EQinfo_{split}.npy", allow_pickle=True),
        station_disp_files=station_disp_files,
        station_vel_files=station_vel_files,
        loc_disp_paths=np.load(split_dir / f"X_{split}_loc_disp.npy", allow_pickle=True),
        loc_vel_paths=np.load(split_dir / f"X_{split}_loc_vel.npy", allow_pickle=True),
        time_length=data_config.time_length,
        n_stations=data_config.n_stations,
        min_removed_stations=data_config.min_removed_stations,
        min_station_dist=data_config.min_station_dist,
        wavetype=data_config.wavetype,
        final_size=data_config.final_size,
        slip_area_classes=data_config.slip_area_classes,
        shuffle_stations=data_config.shuffle_stations and split == "train",
    )
