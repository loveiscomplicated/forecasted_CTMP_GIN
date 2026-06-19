from __future__ import annotations

import torch

from src.analysis.top3_fallback_ablation import (
    _apply_fallback_strategy,
    _apply_top3_admission_fallback,
    _compute_train_mode_values,
    _extract_experiment_id,
    _resolve_variant_head_pairs,
)


class _DummyDataset:
    def __init__(self) -> None:
        self.rows = [
            torch.tensor([1, 2, 4, 0, 1, 1, 0, 2, 1, 3, 0], dtype=torch.long),
            torch.tensor([3, 1, 2, 2, 0, 0, 2, 1, 3, 2, 0], dtype=torch.long),
            torch.tensor([1, 2, 2, 1, 1, 4, 1, 0, 1, 2, 1], dtype=torch.long),
        ]
        self.col_info = (
            [
                "SERVICES",
                "SUB1",
                "FREQ_ATND_SELF_HELP",
                "LIVARAG",
                "FREQ1",
                "SERVICES_D",
                "SUB1_D",
                "FREQ_ATND_SELF_HELP_D",
                "LIVARAG_D",
                "FREQ1_D",
                "OTHER_D",
            ],
            [4, 3, 5, 2, 4, 4, 3, 5, 2, 4, 2],
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9, 10],
        )

    def __getitem__(self, index: int):
        return self.rows[index], 0, 0

    def __len__(self) -> int:
        return len(self.rows)


def test_extract_experiment_id_parses_downstream_name() -> None:
    assert _extract_experiment_id("ctmp_gin_joint_fresh_id38") == 38


def test_apply_top3_admission_fallback_overwrites_only_target_heads() -> None:
    dataset = _DummyDataset()
    x_cache = torch.stack(dataset.rows).clone()
    soft_cache = {
        "head_names": ["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D", "LIVARAG_D", "FREQ1_D", "OTHER_D"],
        "soft_head_names": ["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D", "LIVARAG_D", "FREQ1_D", "OTHER_D"],
        "metadata": {},
        "heads": {
            "SERVICES_D": {
                "hard": torch.zeros(3, dtype=torch.long),
                "probs": torch.zeros((3, 4), dtype=torch.float32),
                "logits": torch.zeros((3, 4), dtype=torch.float32),
                "mask": torch.zeros(3, dtype=torch.bool),
            },
            "SUB1_D": {
                "hard": torch.zeros(3, dtype=torch.long),
                "probs": torch.zeros((3, 3), dtype=torch.float32),
                "logits": torch.zeros((3, 3), dtype=torch.float32),
                "mask": torch.zeros(3, dtype=torch.bool),
            },
            "FREQ_ATND_SELF_HELP_D": {
                "hard": torch.zeros(3, dtype=torch.long),
                "probs": torch.zeros((3, 5), dtype=torch.float32),
                "logits": torch.zeros((3, 5), dtype=torch.float32),
                "mask": torch.zeros(3, dtype=torch.bool),
            },
            "LIVARAG_D": {
                "hard": torch.zeros(3, dtype=torch.long),
                "probs": torch.zeros((3, 2), dtype=torch.float32),
                "logits": torch.zeros((3, 2), dtype=torch.float32),
                "mask": torch.zeros(3, dtype=torch.bool),
            },
            "FREQ1_D": {
                "hard": torch.zeros(3, dtype=torch.long),
                "probs": torch.zeros((3, 4), dtype=torch.float32),
                "logits": torch.zeros((3, 4), dtype=torch.float32),
                "mask": torch.zeros(3, dtype=torch.bool),
            },
            "OTHER_D": {
                "hard": torch.tensor([1, 1, 1], dtype=torch.long),
                "probs": torch.full((3, 2), 0.5, dtype=torch.float32),
                "logits": torch.zeros((3, 2), dtype=torch.float32),
                "mask": torch.ones(3, dtype=torch.bool),
            },
        },
    }

    _apply_top3_admission_fallback(dataset, x_cache, soft_cache, [0, 1])

    assert x_cache[:, 5].tolist() == [1, 3, 4]
    assert x_cache[:, 6].tolist() == [2, 1, 1]
    assert x_cache[:, 7].tolist() == [4, 2, 0]
    assert x_cache[:, 8].tolist() == [1, 3, 1]
    assert x_cache[:, 9].tolist() == [3, 2, 2]
    assert x_cache[:, 10].tolist() == [0, 0, 1]

    services_probs = soft_cache["heads"]["SERVICES_D"]["probs"]
    sub1_probs = soft_cache["heads"]["SUB1_D"]["probs"]
    freq_probs = soft_cache["heads"]["FREQ_ATND_SELF_HELP_D"]["probs"]
    other_probs = soft_cache["heads"]["OTHER_D"]["probs"]

    assert torch.allclose(services_probs[0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(services_probs[1], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(sub1_probs[0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(sub1_probs[1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(freq_probs[0], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(freq_probs[1], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(other_probs, torch.full((3, 2), 0.5))


def test_compute_train_mode_values_uses_train_core_only() -> None:
    dataset = _DummyDataset()
    head_pairs = [
        ("SERVICES_D", "SERVICES"),
        ("SUB1_D", "SUB1"),
        ("LIVARAG_D", "LIVARAG"),
    ]
    modes = _compute_train_mode_values(dataset, [0, 2], head_pairs)

    assert modes == {
        "SERVICES_D": 1,
        "SUB1_D": 0,
        "LIVARAG_D": 1,
    }


def test_apply_train_mode_fallback_overwrites_only_target_heads() -> None:
    dataset = _DummyDataset()
    x_cache = torch.stack(dataset.rows).clone()
    soft_cache = {
        "head_names": ["SERVICES_D", "LIVARAG_D", "OTHER_D"],
        "soft_head_names": ["SERVICES_D", "LIVARAG_D", "OTHER_D"],
        "metadata": {},
        "heads": {
            "SERVICES_D": {
                "hard": torch.zeros(3, dtype=torch.long),
                "probs": torch.zeros((3, 4), dtype=torch.float32),
                "logits": torch.zeros((3, 4), dtype=torch.float32),
                "mask": torch.zeros(3, dtype=torch.bool),
            },
            "LIVARAG_D": {
                "hard": torch.zeros(3, dtype=torch.long),
                "probs": torch.zeros((3, 2), dtype=torch.float32),
                "logits": torch.zeros((3, 2), dtype=torch.float32),
                "mask": torch.zeros(3, dtype=torch.bool),
            },
            "OTHER_D": {
                "hard": torch.tensor([1, 1, 1], dtype=torch.long),
                "probs": torch.full((3, 2), 0.5, dtype=torch.float32),
                "logits": torch.zeros((3, 2), dtype=torch.float32),
                "mask": torch.ones(3, dtype=torch.bool),
            },
        },
    }

    _apply_fallback_strategy(
        dataset,
        x_cache,
        soft_cache,
        [0, 1, 2],
        [("SERVICES_D", "SERVICES"), ("LIVARAG_D", "LIVARAG")],
        strategy="train_mode",
        train_mode_values={"SERVICES_D": 1, "LIVARAG_D": 1},
    )

    assert x_cache[:, 5].tolist() == [1, 1, 1]
    assert x_cache[:, 8].tolist() == [1, 1, 1]
    assert x_cache[:, 10].tolist() == [0, 0, 1]
    assert torch.allclose(soft_cache["heads"]["SERVICES_D"]["probs"][0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(soft_cache["heads"]["LIVARAG_D"]["probs"][0], torch.tensor([0.0, 1.0]))


def test_resolve_variant_head_pairs_top5_uses_diagnostics_ranking() -> None:
    pairs = _resolve_variant_head_pairs("top5")
    assert [pair[0] for pair in pairs[:5]] == [
        "SERVICES_D",
        "FREQ_ATND_SELF_HELP_D",
        "SUB1_D",
        "FREQ1_D",
        "LIVARAG_D",
    ]
