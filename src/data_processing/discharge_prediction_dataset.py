import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.data_processing.canonical_teds import build_canonical_teds_bundle


class DischargePredictionDataset(Dataset):
    """Canonical view for discharge-side predictor targets."""

    def __init__(self, root: str, do_preprocess: bool = False, include_los_in_targets: bool = True):
        super().__init__()
        self.root = root
        bundle = build_canonical_teds_bundle(
            root=root,
            binary=True,
            ig_label=False,
            remove_los=True,
            do_preprocess=do_preprocess,
            admission_only=False,
        )

        self.schema_metadata = bundle.schema_metadata()
        self.ad_col_names = list(bundle.admission_col_names)
        self.ad_col_dims = list(bundle.admission_col_dims)
        self.target_col_names = list(bundle.discharge_target_col_names)
        self.target_col_dims = list(bundle.discharge_target_col_dims)
        self.raw_row_index = bundle.raw_row_index.reset_index(drop=True)
        self.caseid_series = None if bundle.caseid_series is None else bundle.caseid_series.reset_index(drop=True)
        self.x = bundle.x_tensor[:, bundle.col_info[2]]

        target_tensors = [
            torch.tensor(bundle.encoded_feature_df[name].to_numpy(), dtype=torch.long)
            for name in self.target_col_names
        ]
        if include_los_in_targets:
            self.target_col_names.append("LOS")
            self.target_col_dims.append(int(bundle.los_num_classes))
            target_tensors.append(bundle.los_encoded_tensor.long())
        self.y = torch.stack(target_tensors, dim=1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def split_discharge_dataset(
    dataset: DischargePredictionDataset,
    batch_size: int,
    ratio=(0.7, 0.15, 0.15),
    seed: int = 42,
    num_workers: int = 0,
):
    assert abs(sum(ratio) - 1.0) < 1e-6, "ratio must sum to 1.0"
    np.random.seed(seed)
    torch.manual_seed(seed)

    N = len(dataset)
    indices = np.random.permutation(N)
    n_train = int(N * ratio[0])
    n_val = int(N * ratio[1])

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train : n_train + n_val].tolist()
    test_idx = indices[n_train + n_val :].tolist()

    print(f"Train Set Size: {len(train_idx)}")
    print(f"Valid Set Size: {len(val_idx)}")
    print(f"Test Set Size:  {len(test_idx)}")

    _persistent = num_workers > 0
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=_persistent,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=_persistent,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=_persistent,
    )
    return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)
