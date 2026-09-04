# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the frozen robust incoherent normalizer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from quest_transient_backend.fblock_power import (
    INTEGRATION_SAMPLES,
    LOGICAL_STREAMS,
    OUTPUT_CHANNELS,
    PowerIntegrationResult,
)
from quest_transient_backend.incoherent_normalizer import (
    MAD_SCALE_FACTOR,
    CalibrationProvenance,
    IncoherentNormalizer,
    NormalizationStatus,
    RobustCalibration,
    fit_robust_calibration,
)


HASHES = {
    "layout_hash": "a" * 64,
    "frequency_hash": "b" * 64,
    "detector_config_hash": "c" * 64,
    "input_identifier": "history-input-1",
}
HISTORY_FRAMES = 7
HISTORY_STOP = HISTORY_FRAMES * INTEGRATION_SAMPLES
SCIENCE_FIRST = HISTORY_STOP
VALID_UNTIL = 10_000_000


def _masks(
    stream_count: int = LOGICAL_STREAMS,
    *,
    channel_count: int = OUTPUT_CHANNELS,
) -> tuple[np.ndarray, np.ndarray]:
    stream = np.zeros(LOGICAL_STREAMS, dtype=bool)
    stream[:stream_count] = True
    channel = np.zeros(OUTPUT_CHANNELS, dtype=bool)
    channel[:channel_count] = True
    return stream, channel


def _fit(
    stream_mask: np.ndarray | None = None,
    channel_mask: np.ndarray | None = None,
    *,
    history_first: int = 0,
    history_stop: int = HISTORY_STOP,
    valid_until: int = VALID_UNTIL,
) -> RobustCalibration:
    if stream_mask is None or channel_mask is None:
        stream_mask, channel_mask = _masks()
    pattern = np.asarray([-3, -2, -1, 0, 1, 2, 3], dtype=np.float64)
    location = (
        100.0
        + np.arange(LOGICAL_STREAMS, dtype=np.float64)[:, None]
        + np.arange(OUTPUT_CHANNELS, dtype=np.float64)[None, :] / 1000.0
    )
    history = location[None, :, :] + 2.0 * pattern[:, None, None]
    return RobustCalibration.fit(
        history,
        history_first_sample_index=history_first,
        history_stop_sample_exclusive=history_stop,
        valid_until_sample_exclusive=valid_until,
        stream_mask=stream_mask,
        channel_mask=channel_mask,
        **HASHES,
    )


def _result(
    power: np.ndarray,
    *,
    first: int = SCIENCE_FIRST,
    weights: np.ndarray | None = None,
    gaps: np.ndarray | None = None,
) -> PowerIntegrationResult:
    n_time = int(power.shape[0])
    if weights is None:
        weights = np.full(OUTPUT_CHANNELS, 8.0, dtype=np.float64)
    if gaps is None:
        gaps = np.zeros(n_time, dtype=bool)
    first_array = first + np.arange(n_time, dtype=np.int64) * INTEGRATION_SAMPLES
    return PowerIntegrationResult(
        power=np.asarray(power, dtype=np.float64),
        valid_weight=weights,
        first_frame_sample_index=first_array,
        stop_sample_exclusive=first_array + INTEGRATION_SAMPLES,
        gap_before=gaps,
    )


def _science_for_residual(
    calibration: RobustCalibration,
    residual: np.ndarray,
    *,
    first: int = SCIENCE_FIRST,
    weights: np.ndarray | None = None,
    gaps: np.ndarray | None = None,
) -> PowerIntegrationResult:
    power = calibration.location[None, :, :] + calibration.scale[None, :, :] * residual
    return _result(power, first=first, weights=weights, gaps=gaps)


def _apply(
    calibration: RobustCalibration,
    result: PowerIntegrationResult,
    **overrides: object,
):
    provenance = dict(HASHES)
    provenance.update(overrides)
    return IncoherentNormalizer(calibration).apply(result, **provenance)


def test_fit_uses_median_and_14826_mad_and_freezes_provenance() -> None:
    stream_mask, channel_mask = _masks(2, channel_count=4)
    calibration = _fit(stream_mask, channel_mask)

    assert calibration.ready
    np.testing.assert_allclose(calibration.location[:, :4], 100.0 + np.arange(8)[:, None] + np.arange(4)[None, :] / 1000.0)
    np.testing.assert_allclose(calibration.scale[:2, :4], 4.0 * MAD_SCALE_FACTOR)
    assert calibration.provenance.history_range == (0, HISTORY_STOP)
    assert calibration.provenance.valid_until_sample_exclusive == VALID_UNTIL
    with pytest.raises(ValueError):
        calibration.location[0, 0] = 0.0
    with pytest.raises(ValueError):
        calibration.scale[0, 0] = 1.0
    with pytest.raises(ValueError):
        calibration.stream_mask[0] = False
    with pytest.raises(ValueError):
        calibration.channel_mask[0] = False


@pytest.mark.parametrize(
    "field,value",
    [
        ("layout_hash", "not-a-hash"),
        ("frequency_hash", "x"),
        ("detector_config_hash", "y"),
    ],
)
def test_provenance_rejects_non_sha256_hashes(field: str, value: str) -> None:
    values = dict(HASHES)
    values[field] = value
    with pytest.raises(ValueError):
        CalibrationProvenance(
            history_first_sample_index=0,
            history_stop_sample_exclusive=HISTORY_STOP,
            valid_until_sample_exclusive=VALID_UNTIL,
            **values,
        )


def test_fit_rejects_history_range_not_equal_to_contiguous_dump_count() -> None:
    stream_mask, channel_mask = _masks()
    history = np.ones((HISTORY_FRAMES, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    with pytest.raises(ValueError):
        RobustCalibration.fit(
            history,
            history_first_sample_index=0,
            history_stop_sample_exclusive=100,
            valid_until_sample_exclusive=VALID_UNTIL,
            stream_mask=stream_mask,
            channel_mask=channel_mask,
            **HASHES,
        )


@pytest.mark.parametrize("active_streams", [1, 2, 4, 8])
def test_incoherent_sum_scales_as_sqrt_active_stream_count(active_streams: int) -> None:
    stream_mask, channel_mask = _masks(active_streams)
    calibration = _fit(stream_mask, channel_mask)
    residual = np.zeros((2, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    residual[:, :active_streams, :] = 1.0
    output = _apply(calibration, _science_for_residual(calibration, residual))

    assert output.status is NormalizationStatus.ENABLED
    np.testing.assert_allclose(output.science, np.full((2, OUTPUT_CHANNELS), np.sqrt(active_streams)))
    np.testing.assert_allclose(output.normalized_residual[:, :active_streams, :], 1.0)
    np.testing.assert_allclose(output.monitor[:, :active_streams], 1.0)
    np.testing.assert_array_equal(output.monitor[:, active_streams:], 0.0)
    np.testing.assert_array_equal(output.channel_valid, np.ones(OUTPUT_CHANNELS, dtype=bool))


def test_stream_and_channel_masks_and_zero_valid_weight_disable_contributions() -> None:
    stream_mask, channel_mask = _masks(2, channel_count=4)
    calibration = _fit(stream_mask, channel_mask)
    residual = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    weights = np.full(OUTPUT_CHANNELS, 8.0, dtype=np.float64)
    weights[1] = 0.0
    output = _apply(calibration, _science_for_residual(calibration, residual, weights=weights))

    expected_valid = np.zeros(OUTPUT_CHANNELS, dtype=bool)
    expected_valid[[0, 2, 3]] = True
    np.testing.assert_array_equal(output.channel_valid, expected_valid)
    np.testing.assert_allclose(output.science[0, expected_valid], np.sqrt(2.0))
    np.testing.assert_array_equal(output.science[0, ~expected_valid], 0.0)
    np.testing.assert_allclose(output.monitor[0, :2], 1.0)
    np.testing.assert_array_equal(output.monitor[0, 2:], 0.0)
    np.testing.assert_array_equal(output.normalized_residual[0, :, 1], 0.0)


def test_monitor_is_mean_residual_over_active_channels_and_is_finite() -> None:
    stream_mask, channel_mask = _masks(2, channel_count=4)
    calibration = _fit(stream_mask, channel_mask)
    residual = np.zeros((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    residual[0, 0, :4] = [1.0, 2.0, 3.0, 4.0]
    residual[0, 1, :4] = [-2.0, 0.0, 2.0, 4.0]
    output = _apply(calibration, _science_for_residual(calibration, residual))

    np.testing.assert_allclose(output.monitor[0, :2], [2.5, 1.0])
    np.testing.assert_array_equal(output.monitor[0, 2:], 0.0)
    expected = (residual[0, 0, :4] + residual[0, 1, :4]) / np.sqrt(2.0)
    np.testing.assert_allclose(output.science[0, :4], expected)
    assert np.all(np.isfinite(output.monitor))


def test_ranges_and_gap_flags_are_propagated_unchanged() -> None:
    calibration = _fit()
    residual = np.ones((2, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    gaps = np.asarray([True, False], dtype=bool)
    input_result = _science_for_residual(calibration, residual, first=SCIENCE_FIRST, gaps=gaps)
    output = _apply(calibration, input_result)

    np.testing.assert_array_equal(output.first_frame_sample_index, input_result.first_frame_sample_index)
    np.testing.assert_array_equal(output.stop_sample_exclusive, input_result.stop_sample_exclusive)
    np.testing.assert_array_equal(output.gap_before, input_result.gap_before)


def test_history_is_past_only_and_science_changes_cannot_leak_into_calibration() -> None:
    calibration = _fit()
    location_before = calibration.location.copy()
    scale_before = calibration.scale.copy()
    residual = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    science = _science_for_residual(calibration, residual)
    changed = _science_for_residual(calibration, residual * 3.0)
    first_output = _apply(calibration, science)
    second_output = _apply(calibration, changed)

    np.testing.assert_array_equal(calibration.location, location_before)
    np.testing.assert_array_equal(calibration.scale, scale_before)
    assert not np.array_equal(first_output.science, second_output.science)


def test_normalizer_consumes_power_only_and_has_no_voltage_phase_or_hv_cross_term() -> None:
    calibration = _fit(*_masks(2, channel_count=4))
    first_power = np.zeros((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    first_power[:, :2, :4] = 1.0
    second_power = first_power.copy()
    first = _apply(calibration, _science_for_residual(calibration, first_power))
    second = _apply(calibration, _science_for_residual(calibration, second_power))
    np.testing.assert_array_equal(first.science, second.science)
    np.testing.assert_array_equal(first.normalized_residual, second.normalized_residual)


def test_history_science_overlap_is_disabled() -> None:
    calibration = _fit()
    residual = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    output = _apply(calibration, _science_for_residual(calibration, residual, first=HISTORY_STOP - 1))
    assert output.status is NormalizationStatus.DISABLED
    assert output.disabled_reason == "history_science_overlap"
    np.testing.assert_array_equal(output.science, 0.0)


def test_expired_calibration_is_disabled() -> None:
    calibration = _fit(valid_until=HISTORY_STOP + INTEGRATION_SAMPLES - 1)
    residual = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    output = _apply(calibration, _science_for_residual(calibration, residual))
    assert output.status is NormalizationStatus.DISABLED
    assert output.disabled_reason == "calibration_expired"


@pytest.mark.parametrize("field", ["layout_hash", "frequency_hash", "detector_config_hash", "input_identifier"])
def test_provenance_mismatch_is_disabled(field: str) -> None:
    calibration = _fit()
    residual = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    wrong = "0" * 64 if field != "input_identifier" else "wrong"
    output = _apply(calibration, _science_for_residual(calibration, residual), **{field: wrong})
    assert output.status is NormalizationStatus.DISABLED
    assert output.disabled_reason == "provenance_mismatch"


def test_missing_calibration_and_missing_provenance_are_explicitly_disabled() -> None:
    residual = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    calibration = _fit()
    science = _science_for_residual(calibration, residual)
    missing = IncoherentNormalizer(None).apply(
        science,
        layout_hash=HASHES["layout_hash"],
        frequency_hash=HASHES["frequency_hash"],
        detector_config_hash=HASHES["detector_config_hash"],
        input_identifier=HASHES["input_identifier"],
    )
    assert missing.status is NormalizationStatus.DISABLED
    assert missing.disabled_reason == "missing_calibration"
    missing_provenance = IncoherentNormalizer(calibration).apply(science)
    assert missing_provenance.status is NormalizationStatus.DISABLED
    assert missing_provenance.disabled_reason.startswith("missing_provenance:")


def test_not_ready_history_and_nonpositive_active_scale_disable() -> None:
    stream_mask, channel_mask = _masks(1, channel_count=1)
    one_row = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    calibration = RobustCalibration.fit(
        one_row,
        history_first_sample_index=0,
        history_stop_sample_exclusive=INTEGRATION_SAMPLES,
        valid_until_sample_exclusive=VALID_UNTIL,
        stream_mask=stream_mask,
        channel_mask=channel_mask,
        **HASHES,
    )
    assert not calibration.ready
    science = _result(np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64))
    output = _apply(calibration, science)
    assert output.status is NormalizationStatus.DISABLED
    assert output.disabled_reason == "nonpositive_active_scale"


def test_all_zero_stream_mask_is_not_ready_and_disables_detector() -> None:
    calibration = _fit(*_masks(0))
    assert not calibration.ready
    output = _apply(calibration, _result(np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS))))
    assert output.status is NormalizationStatus.DISABLED
    assert output.disabled_reason == "no_active_stream_or_channel"


def test_inactive_zero_scales_are_allowed_but_never_contribute() -> None:
    stream_mask, channel_mask = _masks(1, channel_count=1)
    location = np.ones((LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    scale = np.ones_like(location)
    scale[1:, :] = 0.0
    calibration = RobustCalibration(
        location=location,
        scale=scale,
        stream_mask=stream_mask,
        channel_mask=channel_mask,
        provenance=CalibrationProvenance(
            history_first_sample_index=0,
            history_stop_sample_exclusive=HISTORY_STOP,
            valid_until_sample_exclusive=VALID_UNTIL,
            **HASHES,
        ),
        ready=True,
    )
    residual = np.zeros((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    residual[:, 0, 0] = 2.0
    power = calibration.location[None] + residual * calibration.scale[None]
    output = _apply(calibration, _result(power))
    assert output.enabled
    assert output.science[0, 0] == 2.0
    assert np.all(output.science[0, 1:] == 0.0)


def test_input_and_calibration_are_not_mutated() -> None:
    calibration = _fit()
    residual = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    science = _science_for_residual(calibration, residual)
    power_before = science.power.copy()
    location_before = calibration.location.copy()
    scale_before = calibration.scale.copy()
    output = _apply(calibration, science)

    np.testing.assert_array_equal(science.power, power_before)
    np.testing.assert_array_equal(calibration.location, location_before)
    np.testing.assert_array_equal(calibration.scale, scale_before)
    with pytest.raises(ValueError):
        output.science[0, 0] = 0.0


@pytest.mark.parametrize(
    "bad_history",
    [
        np.ones((0, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64),
        np.ones((1, LOGICAL_STREAMS - 1, OUTPUT_CHANNELS), dtype=np.float64),
        np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float32),
        -np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64),
    ],
)
def test_fit_rejects_empty_shape_dtype_and_negative_history(bad_history: np.ndarray) -> None:
    stream_mask, channel_mask = _masks()
    with pytest.raises((ValueError, TypeError)):
        RobustCalibration.fit(
            bad_history,
            history_first_sample_index=0,
            history_stop_sample_exclusive=HISTORY_STOP,
            valid_until_sample_exclusive=VALID_UNTIL,
            stream_mask=stream_mask,
            channel_mask=channel_mask,
            **HASHES,
        )


def test_fit_rejects_nonfinite_history_and_nonboolean_masks() -> None:
    stream_mask, channel_mask = _masks()
    history = np.ones((2, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    history[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        RobustCalibration.fit(
            history,
            history_first_sample_index=0,
            history_stop_sample_exclusive=HISTORY_STOP,
            valid_until_sample_exclusive=VALID_UNTIL,
            stream_mask=stream_mask,
            channel_mask=channel_mask,
            **HASHES,
        )
    with pytest.raises(ValueError):
        _fit(np.ones(LOGICAL_STREAMS, dtype=np.int8), channel_mask)
    with pytest.raises(ValueError):
        _fit(stream_mask, np.ones(OUTPUT_CHANNELS, dtype=np.int8))


@pytest.mark.parametrize(
    "bad_power",
    [
        np.ones((1, LOGICAL_STREAMS - 1, OUTPUT_CHANNELS), dtype=np.float64),
        np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float32),
        np.full((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), np.nan, dtype=np.float64),
        -np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64),
    ],
)
def test_apply_invalid_power_returns_disabled_without_raising(bad_power: np.ndarray) -> None:
    calibration = _fit()
    bad_result = SimpleNamespace(
        power=bad_power,
        valid_weight=np.ones(OUTPUT_CHANNELS, dtype=np.float64) * 8.0,
        first_frame_sample_index=np.asarray([SCIENCE_FIRST], dtype=np.int64),
        stop_sample_exclusive=np.asarray([SCIENCE_FIRST + INTEGRATION_SAMPLES], dtype=np.int64),
        gap_before=np.asarray([False], dtype=bool),
    )
    output = _apply(calibration, bad_result)
    assert output.status is NormalizationStatus.DISABLED
    assert output.science.shape == (1, OUTPUT_CHANNELS)
    assert np.all(np.isfinite(output.science))


def test_apply_rejects_valid_weight_shape_dtype_nonfinite_and_out_of_range() -> None:
    calibration = _fit()
    power = np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    base = {
        "power": power,
        "first_frame_sample_index": np.asarray([SCIENCE_FIRST], dtype=np.int64),
        "stop_sample_exclusive": np.asarray([SCIENCE_FIRST + INTEGRATION_SAMPLES], dtype=np.int64),
        "gap_before": np.asarray([False], dtype=bool),
    }
    for weight in (
        np.ones(OUTPUT_CHANNELS - 1, dtype=np.float64),
        np.ones(OUTPUT_CHANNELS, dtype=np.float32),
        np.full(OUTPUT_CHANNELS, np.nan, dtype=np.float64),
        np.full(OUTPUT_CHANNELS, 9.0, dtype=np.float64),
        np.full(OUTPUT_CHANNELS, 1.5, dtype=np.float64),
    ):
        output = _apply(calibration, SimpleNamespace(valid_weight=weight, **base))
        assert output.status is NormalizationStatus.DISABLED


def test_all_zero_m1_1_weight_disables_effective_bandwidth() -> None:
    calibration = _fit()
    result = _result(
        np.ones((1, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64),
        weights=np.zeros(OUTPUT_CHANNELS, dtype=np.float64),
    )
    output = _apply(calibration, result)
    assert output.status is NormalizationStatus.DISABLED
    assert output.disabled_reason == "no_effective_bandwidth"
    np.testing.assert_array_equal(output.channel_valid, False)


def test_overlapping_science_ranges_are_disabled() -> None:
    calibration = _fit()
    power = np.ones((2, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    first = np.asarray([SCIENCE_FIRST, SCIENCE_FIRST + 1], dtype=np.int64)
    stop = first + INTEGRATION_SAMPLES
    bad = SimpleNamespace(
        power=power,
        valid_weight=np.full(OUTPUT_CHANNELS, 8.0, dtype=np.float64),
        first_frame_sample_index=first,
        stop_sample_exclusive=stop,
        gap_before=np.asarray([False, False], dtype=bool),
    )
    output = _apply(calibration, bad)
    assert output.status is NormalizationStatus.DISABLED
    assert output.disabled_reason.startswith("invalid_input:")


def test_declared_forward_gap_is_accepted_but_undeclared_hole_is_disabled() -> None:
    calibration = _fit()
    power = np.ones((2, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
    first = np.asarray(
        [SCIENCE_FIRST, SCIENCE_FIRST + 2 * INTEGRATION_SAMPLES], dtype=np.int64
    )
    stop = first + INTEGRATION_SAMPLES
    common = {
        "power": power,
        "valid_weight": np.full(OUTPUT_CHANNELS, 8.0, dtype=np.float64),
        "first_frame_sample_index": first,
        "stop_sample_exclusive": stop,
    }
    declared = _apply(
        calibration,
        SimpleNamespace(gap_before=np.asarray([False, True], dtype=bool), **common),
    )
    assert declared.status is NormalizationStatus.ENABLED
    np.testing.assert_array_equal(declared.gap_before, [False, True])
    undeclared = _apply(
        calibration,
        SimpleNamespace(gap_before=np.asarray([False, False], dtype=bool), **common),
    )
    assert undeclared.status is NormalizationStatus.DISABLED
    assert undeclared.disabled_reason.startswith("invalid_input:")


def test_functional_fit_alias_and_empty_result_are_supported() -> None:
    history = np.stack(
        [
            np.ones((LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64),
            np.full((LOGICAL_STREAMS, OUTPUT_CHANNELS), 2.0, dtype=np.float64),
        ]
    )
    calibration = fit_robust_calibration(
        history,
        history_first_sample_index=0,
        history_stop_sample_exclusive=2 * INTEGRATION_SAMPLES,
        valid_until_sample_exclusive=VALID_UNTIL,
        stream_mask=np.ones(LOGICAL_STREAMS, dtype=bool),
        channel_mask=np.ones(OUTPUT_CHANNELS, dtype=bool),
        **HASHES,
    )
    assert calibration.ready
    empty = PowerIntegrationResult(
        power=np.empty((0, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64),
        valid_weight=np.full(OUTPUT_CHANNELS, 8.0, dtype=np.float64),
        first_frame_sample_index=np.empty(0, dtype=np.int64),
        stop_sample_exclusive=np.empty(0, dtype=np.int64),
        gap_before=np.empty(0, dtype=bool),
    )
    output = _apply(calibration, empty)
    assert output.enabled
    assert output.science.shape == (0, OUTPUT_CHANNELS)
    assert output.monitor.shape == (0, LOGICAL_STREAMS)
    np.testing.assert_array_equal(output.channel_valid, np.ones(OUTPUT_CHANNELS, dtype=bool))
