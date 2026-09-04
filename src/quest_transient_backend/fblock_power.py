# SPDX-License-Identifier: Apache-2.0
"""Pure-NumPy stateful power reference for production-shape FBlocks.

The input contract is exactly ``complex[time, 4096, 8]``.  Each logical
stream is squared independently in the voltage domain; H and V voltages are
never added.  Static ``valid_mask[channel]`` values mean *valid fine channel*.
Every complete 64-PFB-frame integration is reduced in consecutive groups of
eight fine channels and divided by ``64 * valid_fine_channel_count`` for that
group.  A group with no valid fine channels emits finite zero power and a zero
weight.

``FBlockPowerIntegrator.ingest`` accepts arbitrary time chunk boundaries.  It
keeps at most 63 power frames as carry, emits only complete integrations, and
tracks ADC-sample coordinates in units of 4096 samples per PFB frame.  An
explicit ``gap_before=True`` discards carry and arms ``GAP_BEFORE`` on the
next complete output; pre-gap and post-gap frames are never combined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FFT_SIZE = 4096
LOGICAL_STREAMS = 8
OUTPUT_CHANNELS = 512
INTEGRATION_FRAMES = 64
INTEGRATION_SAMPLES = FFT_SIZE * INTEGRATION_FRAMES
CHANNEL_REDUCTION_FACTOR = 8


@dataclass(frozen=True)
class PowerIntegrationResult:
    """One state update's complete integrations.

    ``power`` has shape ``(n_output, 8, 512)`` and float64 dtype.  The static
    ``valid_weight`` has shape ``(512,)``: each value is the number of valid
    fine channels in that group, and is broadcast over output time and stream
    axes.  ``first_frame_sample_index`` and
    ``stop_sample_exclusive`` have shape ``(n_output,)`` and define half-open
    ADC-sample ranges.  ``gap_before`` marks the first complete integration
    after an explicit input gap.
    """

    power: np.ndarray
    valid_weight: np.ndarray
    first_frame_sample_index: np.ndarray
    stop_sample_exclusive: np.ndarray
    gap_before: np.ndarray

    def __post_init__(self) -> None:
        power = np.asarray(self.power)
        valid_weight = np.asarray(self.valid_weight)
        first = np.asarray(self.first_frame_sample_index)
        stop = np.asarray(self.stop_sample_exclusive)
        gap = np.asarray(self.gap_before)
        if power.ndim != 3 or power.shape[1:] != (LOGICAL_STREAMS, OUTPUT_CHANNELS):
            raise ValueError("power must have shape (n_output, 8, 512)")
        n_output = power.shape[0]
        if power.dtype != np.dtype(np.float64):
            raise ValueError("power must have deterministic float64 dtype")
        if valid_weight.shape != (OUTPUT_CHANNELS,) or valid_weight.dtype != np.dtype(np.float64):
            raise ValueError("valid_weight must have shape (512,) and float64 dtype")
        if first.shape != (n_output,) or stop.shape != (n_output,):
            raise ValueError("sample-range arrays must have shape (n_output,)")
        if first.dtype.kind not in "iu" or stop.dtype.kind not in "iu":
            raise ValueError("sample-range arrays must have integer dtype")
        if gap.shape != (n_output,) or gap.dtype != np.dtype(bool):
            raise ValueError("gap_before must have shape (n_output,) and bool dtype")
        if not np.all(np.isfinite(power)) or not np.all(np.isfinite(valid_weight)):
            raise ValueError("power and valid_weight must be finite")
        if np.any(valid_weight < 0.0) or np.any(valid_weight > CHANNEL_REDUCTION_FACTOR):
            raise ValueError("valid_weight must contain valid fine-channel counts")
        if np.any(stop != first + INTEGRATION_SAMPLES):
            raise ValueError("sample ranges must span exactly one 64-frame integration")
        for array in (power, valid_weight, first, stop, gap):
            array.setflags(write=False)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "valid_weight", valid_weight)
        object.__setattr__(self, "first_frame_sample_index", first)
        object.__setattr__(self, "stop_sample_exclusive", stop)
        object.__setattr__(self, "gap_before", gap)

    @property
    def first_sample(self) -> np.ndarray:
        """Alias for the first ADC sample of each emitted integration."""

        return self.first_frame_sample_index

    @property
    def sample_count(self) -> int:
        """Number of complete integrations emitted by this update."""

        return int(self.power.shape[0])

    @property
    def valid_fine_channel_count(self) -> np.ndarray:
        """Static valid fine-channel count for each output group."""

        return self.valid_weight


def _sample_index(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer ADC sample index")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer ADC sample index")
    return result


def _validate_data(data: Any) -> np.ndarray:
    try:
        array = np.asarray(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FBlock data cannot be converted to an array: {exc}") from exc
    if array.ndim != 3 or array.shape[1:] != (FFT_SIZE, LOGICAL_STREAMS):
        raise ValueError("FBlock data must have exactly shape (time, 4096, 8)")
    if array.shape[0] <= 0:
        raise ValueError("FBlock data must contain at least one time frame")
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("FBlock data must have a complex numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("FBlock data must contain only finite complex values")
    return array


def _validate_mask(valid_mask: Any) -> np.ndarray:
    mask = np.asarray(valid_mask)
    if mask.shape != (FFT_SIZE,):
        raise ValueError("valid_mask must have exactly shape (4096,)")
    if mask.dtype != np.dtype(bool):
        raise ValueError("valid_mask must have boolean dtype; integer masks are ambiguous")
    return mask.copy()


class FBlockPowerIntegrator:
    """Stateful NumPy reference for 64-frame per-stream power integration."""

    def __init__(self, valid_mask: Any):
        self.valid_mask = _validate_mask(valid_mask)
        self.valid_mask.setflags(write=False)
        counts = self.valid_mask.reshape(OUTPUT_CHANNELS, CHANNEL_REDUCTION_FACTOR).sum(axis=1)
        self._valid_weight = counts.astype(np.float64, copy=False)
        self._valid_weight.setflags(write=False)
        self._carry_power: np.ndarray | None = None
        self._carry_start_sample: int | None = None
        self._expected_next_sample: int | None = None
        self._pending_gap = False
        self.frames_ingested = 0
        self.integrations_emitted = 0

    @property
    def carry_frames(self) -> int:
        """Number of power frames retained across calls, always at most 63."""

        return 0 if self._carry_power is None else int(self._carry_power.shape[0])

    @property
    def pending_gap(self) -> bool:
        """Whether the next complete integration must be marked GAP_BEFORE."""

        return self._pending_gap

    @property
    def expected_next_sample(self) -> int | None:
        return self._expected_next_sample

    def reset(self, *, gap_before: bool = True) -> None:
        """Discard partial carry and optionally arm a gap for the next output."""

        if type(gap_before) is not bool:
            raise ValueError("gap_before must be boolean")
        self._carry_power = None
        self._carry_start_sample = None
        self._expected_next_sample = None
        self._pending_gap = gap_before

    def _empty_result(self) -> PowerIntegrationResult:
        return PowerIntegrationResult(
            power=np.empty((0, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64),
            valid_weight=self._valid_weight.copy(),
            first_frame_sample_index=np.empty((0,), dtype=np.int64),
            stop_sample_exclusive=np.empty((0,), dtype=np.int64),
            gap_before=np.empty((0,), dtype=bool),
        )

    def _reduce(self, power_frames: np.ndarray, start_sample: int, count: int) -> PowerIntegrationResult:
        # The mask is applied before reduction.  The output stream axis remains
        # independent; no H/V voltage or cross term is ever formed.
        masked = power_frames.reshape(
            count,
            INTEGRATION_FRAMES,
            OUTPUT_CHANNELS,
            CHANNEL_REDUCTION_FACTOR,
            LOGICAL_STREAMS,
        ) * self.valid_mask.reshape(1, 1, OUTPUT_CHANNELS, CHANNEL_REDUCTION_FACTOR, 1)
        groups = masked.sum(axis=(1, 3), dtype=np.float64)
        groups = np.transpose(groups, (0, 2, 1))
        output = np.zeros((count, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64)
        nonzero = self._valid_weight > 0.0
        if np.any(nonzero):
            output[:, :, nonzero] = groups[:, :, nonzero] / (
                float(INTEGRATION_FRAMES) * self._valid_weight[nonzero][None, None, :]
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("power reduction produced non-finite output")
        first = start_sample + np.arange(count, dtype=np.int64) * INTEGRATION_SAMPLES
        stop = first + INTEGRATION_SAMPLES
        gap = np.zeros((count,), dtype=bool)
        if self._pending_gap:
            gap[0] = True
            self._pending_gap = False
        self.integrations_emitted += count
        return PowerIntegrationResult(
            power=output,
            valid_weight=self._valid_weight.copy(),
            first_frame_sample_index=first,
            stop_sample_exclusive=stop,
            gap_before=gap,
        )

    def ingest(
        self,
        data: Any,
        first_frame_sample_index: Any,
        *,
        gap_before: bool = False,
    ) -> PowerIntegrationResult:
        """Consume one contiguous FBlock chunk and emit complete integrations.

        ``first_frame_sample_index`` is the ADC sample coordinate of the first
        input PFB frame.  A coordinate discontinuity is rejected unless
        ``gap_before=True`` explicitly declares the missing region.
        """

        array = _validate_data(data)
        start_sample = _sample_index(first_frame_sample_index, "first_frame_sample_index")
        if type(gap_before) is not bool:
            raise ValueError("gap_before must be boolean")
        frame_count = int(array.shape[0])
        expected = self._expected_next_sample
        if expected is not None and start_sample != expected and not gap_before:
            raise ValueError(
                "FBlock sample coordinate is discontinuous; set gap_before=True "
                "to declare the gap explicitly"
            )

        # Compute into a fresh float64 array so the caller's complex input is
        # never mutated and overflow/non-finite power cannot be propagated.
        power = np.abs(array.astype(np.complex128, copy=False)) ** 2
        if not np.all(np.isfinite(power)):
            raise ValueError("complex input power is non-finite")
        power = np.asarray(power, dtype=np.float64)

        if gap_before:
            self._carry_power = None
            self._carry_start_sample = None
            self._pending_gap = True
        if self._carry_power is not None:
            if self._carry_start_sample is None:
                raise RuntimeError("internal carry start is missing")
            combined = np.concatenate((self._carry_power, power), axis=0)
            combined_start = self._carry_start_sample
        else:
            combined = power
            combined_start = start_sample

        complete_frames = (combined.shape[0] // INTEGRATION_FRAMES) * INTEGRATION_FRAMES
        output_count = complete_frames // INTEGRATION_FRAMES
        if complete_frames:
            result = self._reduce(combined[:complete_frames], combined_start, output_count)
        else:
            result = self._empty_result()

        remainder = combined[complete_frames:]
        if remainder.shape[0] > 0:
            self._carry_power = remainder.copy()
            self._carry_start_sample = combined_start + complete_frames * FFT_SIZE
        else:
            self._carry_power = None
            self._carry_start_sample = None
        if self.carry_frames >= INTEGRATION_FRAMES:
            raise RuntimeError("internal carry exceeded 63 frames")
        self._expected_next_sample = start_sample + frame_count * FFT_SIZE
        self.frames_ingested += frame_count
        return result

    process = ingest
    ingest_fblock = ingest


PowerIntegrator = FBlockPowerIntegrator
FBlockPowerResult = PowerIntegrationResult
