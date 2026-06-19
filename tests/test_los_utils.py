import torch

from src.models.discharge_predictor.los_utils import (
    expand_coarse_distribution_to_raw_los,
    get_los_coarse_num_classes,
    infer_los_coarse_breakdown_from_cfg,
    los_binning_metadata_dict,
    map_coarse_array_to_raw_los,
    map_los_array_to_coarse_bins,
    map_los_to_coarse_bin,
)


def test_map_los_to_coarse_bin_boundaries() -> None:
    assert map_los_to_coarse_bin(1) == 0
    assert map_los_to_coarse_bin(2) == 1
    assert map_los_to_coarse_bin(7) == 1
    assert map_los_to_coarse_bin(8) == 2
    assert map_los_to_coarse_bin(14) == 2
    assert map_los_to_coarse_bin(15) == 3
    assert map_los_to_coarse_bin(21) == 3
    assert map_los_to_coarse_bin(22) == 4
    assert map_los_to_coarse_bin(28) == 4
    assert map_los_to_coarse_bin(29) == 5
    assert map_los_to_coarse_bin(30) == 5
    assert map_los_to_coarse_bin(31) == 5
    assert map_los_to_coarse_bin(37) == 5


def test_map_los_to_coarse_bin_invalid() -> None:
    try:
        map_los_to_coarse_bin(0)
        raise AssertionError("Expected ValueError for LOS=0")
    except ValueError:
        pass

    try:
        map_los_to_coarse_bin(38)
        raise AssertionError("Expected ValueError for LOS=38")
    except ValueError:
        pass


def test_map_los_to_coarse_bin_breakdown_boundaries() -> None:
    assert map_los_to_coarse_bin(29, breakdown=True) == 5
    assert map_los_to_coarse_bin(31, breakdown=True) == 5
    assert map_los_to_coarse_bin(32, breakdown=True) == 6
    assert map_los_to_coarse_bin(33, breakdown=True) == 6
    assert map_los_to_coarse_bin(34, breakdown=True) == 7
    assert map_los_to_coarse_bin(35, breakdown=True) == 7
    assert map_los_to_coarse_bin(36, breakdown=True) == 8
    assert map_los_to_coarse_bin(37, breakdown=True) == 8


def test_map_los_array_to_coarse_bins_uses_breakdown() -> None:
    raw_los = torch.tensor([1, 29, 32, 34, 36, 37])
    coarse = map_los_array_to_coarse_bins(raw_los, breakdown=True)

    assert coarse.tolist() == [0, 5, 6, 7, 8, 8]


def test_map_coarse_array_to_raw_los_uses_bin_representatives() -> None:
    coarse = torch.arange(6)
    raw_los = map_coarse_array_to_raw_los(coarse)

    assert raw_los.tolist() == [1, 4, 11, 18, 25, 33]


def test_map_coarse_array_to_raw_los_uses_breakdown_representatives() -> None:
    coarse = torch.arange(9)
    raw_los = map_coarse_array_to_raw_los(coarse, breakdown=True)

    assert raw_los.tolist() == [1, 4, 11, 18, 25, 30, 32, 34, 36]


def test_expand_coarse_distribution_to_raw_los_spreads_bin_mass() -> None:
    coarse_probs = torch.zeros(1, 6)
    coarse_probs[0, 5] = 1.0

    raw_probs = expand_coarse_distribution_to_raw_los(coarse_probs)

    assert raw_probs.shape == (1, 37)
    assert torch.isclose(raw_probs.sum(), torch.tensor(1.0))
    assert torch.all(raw_probs[0, :28] == 0)
    assert torch.allclose(raw_probs[0, 28:], torch.full((9,), 1.0 / 9.0))


def test_expand_coarse_distribution_to_raw_los_spreads_breakdown_mass() -> None:
    coarse_probs = torch.zeros(1, 9)
    coarse_probs[0, 8] = 1.0

    raw_probs = expand_coarse_distribution_to_raw_los(coarse_probs, breakdown=True)

    assert raw_probs.shape == (1, 37)
    assert torch.isclose(raw_probs.sum(), torch.tensor(1.0))
    assert torch.all(raw_probs[0, :35] == 0)
    assert torch.allclose(raw_probs[0, 35:], torch.full((2,), 0.5))


def test_breakdown_metadata_and_config_resolution() -> None:
    metadata = los_binning_metadata_dict(breakdown=True)

    assert get_los_coarse_num_classes(breakdown=True) == 9
    assert metadata["num_classes"] == 9
    assert metadata["los_bins"][-4:] == [[29, 31], [32, 33], [34, 35], [36, 37]]
    assert infer_los_coarse_breakdown_from_cfg({"los_coarse_breakdown": True})
    assert infer_los_coarse_breakdown_from_cfg({"num_classes": 9})
