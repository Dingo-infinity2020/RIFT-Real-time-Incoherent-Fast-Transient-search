# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from quest_transient_backend.incoherent_trigger import (
    DMSeriesCalibration,
    IncoherentDMPlan,
    IncoherentTriggerKernel,
    StationPowerCalibration,
    StreamingIncoherentTrigger,
    boxcar_bank,
    dedisperse_incoherent_power,
)


def test_dm_plan_and_dedispersion_recover_reference_time():
    frequency = np.linspace(1.2e9, 1.5e9, 16)
    plan = IncoherentDMPlan.build(frequency, 1e-3, [0.0, 20.0])
    power = np.zeros((128, frequency.size), dtype=np.float32)
    reference_time = 40
    for channel, shift in enumerate(plan.channel_shifts[1]):
        power[reference_time + shift, channel] = 1.0
    series = dedisperse_incoherent_power(power, plan)
    assert np.argmax(series[1]) + plan.left_margin == reference_time
    assert series[1].max() == frequency.size


def test_station_mask_and_channel_mask_are_applied_before_sum():
    calibration = StationPowerCalibration(
        location=np.zeros((3, 4)),
        scale=np.ones((3, 4)),
        station_weights=np.array([1.0, 0.0, 2.0]),
        channel_weights=np.array([1.0, 0.0, 1.0, 1.0]),
        background_dumps=100,
    )
    power = np.ones((5, 3, 4), dtype=np.float32)
    summed = calibration.normalize_and_sum(power)
    np.testing.assert_array_equal(summed[:, 0], 3.0)
    np.testing.assert_array_equal(summed[:, 1], 0.0)


def test_kernel_recovers_injected_dm_and_boxcar_width():
    rng = np.random.default_rng(7)
    frequency = np.linspace(1.2e9, 1.5e9, 32)
    plan = IncoherentDMPlan.build(frequency, 1e-3, [0.0, 10.0, 20.0, 30.0])
    power = rng.normal(10.0, 1.0, (512, 4, 32)).astype(np.float32)
    background = np.ones(512, dtype=bool)
    background[180:260] = False
    power_calibration = StationPowerCalibration.estimate(power, background)
    base_normalized = power_calibration.normalize_and_sum(power)
    base_series = dedisperse_incoherent_power(base_normalized, plan)
    valid_background = background[
        plan.left_margin : plan.left_margin + base_series.shape[1]
    ]
    dm_calibration = DMSeriesCalibration.estimate(
        base_series, plan.dm_trials, valid_background
    )

    injected = power.copy()
    reference_time = 220
    width = 4
    amplitude = 3.0
    for channel, shift in enumerate(plan.channel_shifts[2]):
        injected[reference_time + shift : reference_time + shift + width, :, channel] += amplitude
    result = IncoherentTriggerKernel(
        plan, power_calibration, dm_calibration, widths=(1, 2, 4, 8)
    ).process(injected, first_input_dump=1000)
    dm_index, time_index = np.unravel_index(np.argmax(result.score), result.score.shape)
    assert result.dm_trials[dm_index] == 20.0
    assert time_index + result.first_input_dump == 1000 + reference_time + width // 2
    assert result.best_width[dm_index, time_index] == width


def test_boxcar_rejects_invalid_width_bank():
    with pytest.raises(ValueError, match="include 1"):
        boxcar_bank(np.zeros((2, 10), dtype=np.float32), (2, 4))




class FakeBatch:
    def __init__(
        self, power, first_sample, *, gap=False, stride=16, identity_tag=None
    ):
        self.station_power = power
        self.first_sample = first_sample
        self.dump_count = power.shape[0]
        self.dump_stride_samples = stride
        self.fft_size = power.shape[2]
        self.gap_before_first = gap
        self.identity_builder = (
            (lambda first, count: (identity_tag, first, count))
            if identity_tag is not None
            else (lambda first, count: (first, count))
        )


def _stream_fixture():
    rng = np.random.default_rng(19)
    frequency = np.linspace(1.2e9, 1.5e9, 16)
    plan = IncoherentDMPlan.build(frequency, 1e-3, [0.0, 20.0])
    power = rng.normal(10.0, 1.0, (640, 3, 16)).astype(np.float32)
    calibration = StationPowerCalibration(
        np.full((3, 16), 10.0), np.ones((3, 16)),
        np.ones(3), np.ones(16), 100,
    )
    dm_calibration = DMSeriesCalibration(
        plan.dm_trials, np.zeros(2), np.ones(2), 100
    )
    kernel = IncoherentTriggerKernel(
        plan, calibration, dm_calibration, widths=(1, 2, 4, 8)
    )
    return power, kernel


def test_streaming_chunk_ownership_matches_whole_interval():
    power, kernel = _stream_fixture()
    whole = kernel.process(power)
    state = StreamingIncoherentTrigger(kernel)
    outputs = []
    cursor = 0
    for size in (17, 31, 5, 64, 9, 128, 33, 101, 252):
        result = state.ingest(FakeBatch(power[cursor : cursor + size], cursor * 16))
        if result is not None:
            outputs.append(result)
        cursor += size
    assert cursor == power.shape[0]
    first_dump = outputs[0].first_sample // 16
    stream_score = np.concatenate([item.score for item in outputs], axis=1)
    stream_width = np.concatenate([item.best_width for item in outputs], axis=1)
    whole_start = first_dump - whole.first_input_dump
    np.testing.assert_allclose(
        stream_score, whole.score[:, whole_start : whole_start + stream_score.shape[1]],
        rtol=2e-6, atol=2e-6,
    )
    np.testing.assert_array_equal(
        stream_width, whole.best_width[:, whole_start : whole_start + stream_width.shape[1]]
    )
    coordinates = np.concatenate([
        item.first_sample // 16 + np.arange(item.sample_count) for item in outputs
    ])
    np.testing.assert_array_equal(np.diff(coordinates), 1)
    assert state.max_buffered_dumps < 300
    assert state.ring_capacity >= 512
    assert state.dedispersed_samples < power.shape[0] + 8 * len(outputs)
    assert state.max_dedispersed_samples_per_ingest <= 252 + 7


def test_streaming_gap_resets_and_marks_first_new_output():
    power, kernel = _stream_fixture()
    state = StreamingIncoherentTrigger(kernel)
    assert state.ingest(FakeBatch(power[:200], 0)) is not None
    output = state.ingest(FakeBatch(power[300:500], 300 * 16, gap=True))
    if output is None:
        output = state.ingest(FakeBatch(power[500:], 500 * 16))
    assert output.segment_id == 1
    assert output.gap_before
    assert output.first_sample >= 300 * 16
    assert state.gap_count == 1
    assert output.sample_id_at(0) == (output.first_sample, 16)


def test_streaming_output_routes_to_batch_specific_identity_snapshot():
    power, kernel = _stream_fixture()
    state = StreamingIncoherentTrigger(kernel)
    first = state.ingest(
        FakeBatch(power[:200], 0, identity_tag="first")
    )
    second = state.ingest(
        FakeBatch(power[200:400], 200 * 16, identity_tag="second")
    )
    outputs = [item for item in (first, second) if item is not None]
    routed = []
    for output in outputs:
        for index in range(output.sample_count):
            identity = output.sample_id_at(index)
            if identity[1] >= 200 * 16:
                routed.append(identity)
    assert routed
    assert all(identity[0] == "second" for identity in routed)
