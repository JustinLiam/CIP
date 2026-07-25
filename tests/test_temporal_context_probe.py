import numpy as np
import torch

from scripts.epi_abm.evaluate_temporal_context_probe import (
    centered_relative_change,
    occlude_recency_block,
    truncate_history,
)
from scripts.epi_abm.aggregate_temporal_context_probe import linear_cka


def _history(length=12):
    values = torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
    return {
        "prev_treatments": values.clone(),
        "prev_outputs": (values + 100).clone(),
        "vitals": (values + 200).clone(),
        "current_treatments": (values + 300).clone(),
        "static_features": (values + 400).clone(),
        "active_entries": torch.ones(1, length, 1),
        "sequence_lengths": torch.tensor([length]),
    }


def test_truncate_history_keeps_recent_timesteps_and_updates_length():
    result = truncate_history(_history(), 5)
    assert result["sequence_lengths"].tolist() == [5]
    assert result["prev_treatments"].reshape(-1).tolist() == [7, 8, 9, 10, 11]
    assert result["static_features"].shape[1] == 5
    assert torch.all(result["active_entries"] == 1)


def test_occlusion_replaces_only_encoder_inputs_with_outside_mean():
    history = _history()
    result = occlude_recency_block(history, 1, 3)
    expected_mean = torch.arange(9, dtype=torch.float32).mean()
    assert torch.all(result["prev_treatments"][0, 9:, 0] == expected_mean)
    assert torch.equal(result["current_treatments"], history["current_treatments"])
    assert torch.equal(result["static_features"], history["static_features"])
    assert torch.equal(history["prev_treatments"][0, 9:, 0], torch.tensor([9.0, 10.0, 11.0]))


def test_centered_relative_change_ignores_common_q_offset():
    assert centered_relative_change([1, 2, 3], [11, 12, 13]) < 1e-12
    assert np.isclose(centered_relative_change([1, 2, 3], [1, 3, 5]), 1.0)


def test_linear_cka_is_invariant_to_orthogonal_feature_rotation():
    x = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    assert np.isclose(linear_cka(x, x @ rotation), 1.0)
