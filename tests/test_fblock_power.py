# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the offline FBlock power reference."""

from __future__ import annotations

import numpy as np
import pytest

from quest_transient_backend.fblock_power import (
    CHANNEL_REDUCTION_FACTOR,
    FFT_SIZE,
    INTEGRATION_FRAMES,
    INTEGRATION_SAMPLES,
    LOGICAL_STREAMS,
    OUTPUT_CHANNELS,
    FBlockPowerIntegrator,
)


def _constant(frames: int, amplitudes: object = 1.0, *, dtype: np.dtype = np.dtype(np.complex64)) -> np.ndarray:
    amps = np.broadcast_to(np.asarray(amplitudes, dtype=np.float64), (LOGICAL_STREAMS,))
    data = np.ones((frames, FFT_SIZE, LOGICAL_STREAMS), dtype=np.complex128)
    data *= amps.reshape(1, 1, LOGICAL_STREAMS)
    return data.astype(dtype)


def _stack(results: list[object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    concrete = results
    assert concrete
    return (
        np.concatenate([result.power for result in concrete], axis=0),
        concrete[0].valid_weight,
        np.concatenate([result.first_frame_sample_index for result in concrete], axis=0),
        np.concatenate([result.stop_sample_exclusive for result in concrete], axis=0),
        np.concatenate([result.gap_before for result in concrete], axis=0),
    )


def test_constant_amplitudes_have_per_stream_power_and_expected_axes() -> None:
    amplitudes = np.arange(1.0, LOGICAL_STREAMS + 1.0)
    result = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool)).ingest(
        _constant(INTEGRATION_FRAMES, amplitudes), 0
    )

    assert result.power.shape == (1, LOGICAL_STREAMS, OUTPUT_CHANNELS)
    assert result.power.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        result.power[0], np.broadcast_to(amplitudes[:, None] ** 2, (LOGICAL_STREAMS, OUTPUT_CHANNELS))
    )
    np.testing.assert_array_equal(result.valid_weight, np.full(OUTPUT_CHANNELS, 8.0))
    np.testing.assert_array_equal(result.first_frame_sample_index, [0])
    np.testing.assert_array_equal(result.stop_sample_exclusive, [INTEGRATION_SAMPLES])
    np.testing.assert_array_equal(result.gap_before, [False])
    assert result.first_sample is result.first_frame_sample_index
    assert result.sample_count == 1


def test_chunking_is_invariant_including_one_frame_chunks() -> None:
    data = _constant(128, np.arange(1.0, LOGICAL_STREAMS + 1.0))
    one_shot = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool)).ingest(data, 8192)

    chunked_integrator = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool))
    pieces = []
    for offset in range(data.shape[0]):
        pieces.append(
            chunked_integrator.ingest(data[offset : offset + 1], 8192 + offset * FFT_SIZE)
        )
    power, weight, first, stop, gap = _stack(pieces)

    np.testing.assert_array_equal(power, one_shot.power)
    np.testing.assert_array_equal(weight, one_shot.valid_weight)
    np.testing.assert_array_equal(first, one_shot.first_frame_sample_index)
    np.testing.assert_array_equal(stop, one_shot.stop_sample_exclusive)
    np.testing.assert_array_equal(gap, one_shot.gap_before)
    assert chunked_integrator.carry_frames == 0

    arbitrary = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool))
    pieces = []
    offset = 0
    for size in (7, 31, 1, 25, 64):
        pieces.append(arbitrary.ingest(data[offset : offset + size], 8192 + offset * FFT_SIZE))
        offset += size
    assert offset == data.shape[0]
    power, _, first, stop, gap = _stack(pieces)
    np.testing.assert_array_equal(power, one_shot.power)
    np.testing.assert_array_equal(first, one_shot.first_frame_sample_index)
    np.testing.assert_array_equal(stop, one_shot.stop_sample_exclusive)
    np.testing.assert_array_equal(gap, one_shot.gap_before)


def test_phase_rotation_does_not_change_power() -> None:
    amplitudes = np.arange(1.0, LOGICAL_STREAMS + 1.0)
    base = _constant(INTEGRATION_FRAMES, amplitudes, dtype=np.complex128)
    phase = np.exp(
        1j
        * (
            np.arange(INTEGRATION_FRAMES, dtype=np.float64)[:, None, None]
            + np.arange(FFT_SIZE, dtype=np.float64)[None, :, None] / 13.0
        )
    )
    rotated = base * phase
    mask = np.ones(FFT_SIZE, dtype=bool)
    reference = FBlockPowerIntegrator(mask).ingest(base, 0)
    result = FBlockPowerIntegrator(mask).ingest(rotated, 0)
    np.testing.assert_array_equal(result.power, reference.power)


def test_streams_are_independent_and_voltage_summing_would_differ() -> None:
    # H0 and V0 have equal opposite voltages.  A voltage-domain H+V sum would
    # produce zero, while the contract requires independent powers of one.
    data = _constant(INTEGRATION_FRAMES, 1.0)
    data[:, :, 4] = -1.0 + 0.0j
    result = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool)).ingest(data, 0)

    np.testing.assert_array_equal(result.power[0, 0], np.ones(OUTPUT_CHANNELS))
    np.testing.assert_array_equal(result.power[0, 4], np.ones(OUTPUT_CHANNELS))
    voltage_sum_power = np.abs(data[:, :, 0] + data[:, :, 4]) ** 2
    assert np.all(voltage_sum_power == 0.0)
    assert not np.array_equal(result.power[0, 0], voltage_sum_power.mean(axis=(0, 1)))


def test_partial_mask_normalizes_by_actual_valid_fine_channel_count() -> None:
    mask = np.zeros(FFT_SIZE, dtype=bool)
    mask[:4] = True
    mask[8:10] = True
    data = _constant(INTEGRATION_FRAMES, np.arange(1.0, LOGICAL_STREAMS + 1.0))
    result = FBlockPowerIntegrator(mask).ingest(data, 0)

    np.testing.assert_array_equal(result.valid_weight[:3], [4.0, 2.0, 0.0])
    np.testing.assert_allclose(result.power[0, :, 0], np.arange(1.0, 9.0) ** 2)
    np.testing.assert_allclose(result.power[0, :, 1], np.arange(1.0, 9.0) ** 2)
    np.testing.assert_array_equal(result.power[0, :, 2], np.zeros(LOGICAL_STREAMS))
    assert np.all(np.isfinite(result.power))


def test_sample_ranges_are_half_open_adc_ranges() -> None:
    start = 17 * FFT_SIZE
    data = _constant(128)
    integrator = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool))
    result = integrator.ingest(data, start)
    np.testing.assert_array_equal(
        result.first_frame_sample_index,
        [start, start + INTEGRATION_SAMPLES],
    )
    np.testing.assert_array_equal(
        result.stop_sample_exclusive,
        [start + INTEGRATION_SAMPLES, start + 2 * INTEGRATION_SAMPLES],
    )


def test_gap_discards_partial_carry_and_marks_next_complete_output() -> None:
    mask = np.ones(FFT_SIZE, dtype=bool)
    integrator = FBlockPowerIntegrator(mask)
    pre = _constant(32, 1.0)
    post = _constant(96, 3.0)
    assert integrator.ingest(pre, 0).sample_count == 0
    assert integrator.carry_frames == 32

    gap_start = 10_000_000
    no_output = integrator.ingest(post[:32], gap_start, gap_before=True)
    assert no_output.sample_count == 0
    assert integrator.pending_gap
    assert integrator.carry_frames == 32
    result = integrator.ingest(post[32:], gap_start + 32 * FFT_SIZE)

    assert result.sample_count == 1
    np.testing.assert_array_equal(result.first_frame_sample_index, [gap_start])
    np.testing.assert_array_equal(result.gap_before, [True])
    np.testing.assert_array_equal(result.power, np.full_like(result.power, 9.0))
    assert not integrator.pending_gap
    assert integrator.carry_frames == 32


def test_multiple_gaps_keep_pending_until_the_next_complete_output() -> None:
    integrator = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool))
    assert integrator.ingest(_constant(64), 0).sample_count == 1
    first_gap = 2_000_000
    assert integrator.ingest(_constant(1), first_gap, gap_before=True).sample_count == 0
    second_gap = first_gap + FFT_SIZE
    assert integrator.ingest(_constant(1), second_gap, gap_before=True).sample_count == 0
    assert integrator.pending_gap
    result = integrator.ingest(_constant(63), second_gap + FFT_SIZE)
    np.testing.assert_array_equal(result.gap_before, [True])
    assert not integrator.pending_gap


def test_input_and_mask_are_not_mutated_and_mask_is_static() -> None:
    mask = np.ones(FFT_SIZE, dtype=bool)
    data = _constant(INTEGRATION_FRAMES, 2.0)
    mask_before = mask.copy()
    data_before = data.copy()
    integrator = FBlockPowerIntegrator(mask)
    integrator.ingest(data, 0)
    np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(data, data_before)
    with pytest.raises(ValueError):
        integrator.valid_mask[0] = False


def test_carry_is_bounded_for_every_one_frame_update() -> None:
    integrator = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool))
    for index in range(127):
        result = integrator.ingest(_constant(1, 1.0), index * FFT_SIZE)
        assert result.sample_count in (0, 1)
        assert 0 <= integrator.carry_frames < INTEGRATION_FRAMES
    assert integrator.carry_frames == 63


@pytest.mark.parametrize(
    "bad_data",
    [
        np.ones((1, FFT_SIZE - 1, LOGICAL_STREAMS), dtype=np.complex64),
        np.ones((1, FFT_SIZE, LOGICAL_STREAMS - 1), dtype=np.complex64),
        np.ones((FFT_SIZE, LOGICAL_STREAMS), dtype=np.complex64),
        np.ones((0, FFT_SIZE, LOGICAL_STREAMS), dtype=np.complex64),
        np.ones((1, FFT_SIZE, LOGICAL_STREAMS), dtype=np.float32),
    ],
)
def test_rejects_wrong_shape_or_noncomplex_data(bad_data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool)).ingest(bad_data, 0)


@pytest.mark.parametrize("bad_value", [np.nan + 0j, np.inf + 0j, -np.inf + 0j])
def test_rejects_nonfinite_complex_input(bad_value: complex) -> None:
    data = _constant(1)
    data[0, 0, 0] = bad_value
    with pytest.raises(ValueError):
        FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool)).ingest(data, 0)


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.ones(FFT_SIZE - 1, dtype=bool),
        np.ones(FFT_SIZE, dtype=np.int8),
        np.ones((FFT_SIZE, 1), dtype=bool),
    ],
)
def test_rejects_mask_shape_and_dtype_ambiguity(bad_mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        FBlockPowerIntegrator(bad_mask)


@pytest.mark.parametrize("bad_start", [-1, True, 1.0, np.nan])
def test_rejects_invalid_sample_coordinate(bad_start: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool)).ingest(_constant(1), bad_start)


def test_rejects_implicit_discontinuity_but_accepts_explicit_gap() -> None:
    integrator = FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool))
    assert integrator.ingest(_constant(1), 0).sample_count == 0
    with pytest.raises(ValueError):
        integrator.ingest(_constant(1), 2 * FFT_SIZE)
    assert integrator.ingest(_constant(1), 2 * FFT_SIZE, gap_before=True).sample_count == 0


def test_rejects_nonboolean_gap_flag() -> None:
    with pytest.raises(ValueError):
        FBlockPowerIntegrator(np.ones(FFT_SIZE, dtype=bool)).ingest(
            _constant(1), 0, gap_before=1
        )


def test_reduction_factor_is_explicit_and_fixed() -> None:
    assert CHANNEL_REDUCTION_FACTOR == 8
    assert OUTPUT_CHANNELS * CHANNEL_REDUCTION_FACTOR == FFT_SIZE
