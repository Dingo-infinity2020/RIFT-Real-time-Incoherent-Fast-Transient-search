# SPDX-License-Identifier: Apache-2.0
"""GPU-preserving incoherent DM/boxcar trigger primitives.

The module contains no transport or threshold policy.  It transforms the
station-power view owned by a :class:`DetectorSinkBatch` into an unthresholded
DM/time score grid while retaining the input array backend (NumPy or CuPy).
Calibration is deliberately frozen and external so a candidate cannot train
its own background model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


DISPERSION_DELAY_HZ_SEC = 4.148808e15
_CUPY_DEDISPERSION_KERNEL = None
_CUPY_RING_DEDISPERSION_KERNEL = None
_CUPY_NORMALIZE_RING_KERNEL = None
_CUPY_BOXCAR_KERNEL = None


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy as cp

        return cp
    if module == "numpy":
        return np
    raise TypeError("detector arrays must be NumPy or CuPy arrays")


def _readonly(array, dtype=None):
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


class IdentityRangeBuilder:
    """Resolve delayed sample identities inside retained source ranges.

    Candidate extraction may outlive the trigger batch that first exposed a
    sample.  A resolver therefore snapshots the retained ranges and accepts a
    request only when the complete ``[first_sample, first_sample +
    sample_count)`` interval belongs to one range.  It never clamps or joins
    across a discontinuity.
    """

    def __init__(self, ranges: Sequence[tuple[int, int, Any]]) -> None:
        self._ranges = tuple(ranges)

    def build(self, first_sample: int, sample_count: int):
        first = int(first_sample)
        count = int(sample_count)
        if count <= 0:
            raise ValueError("identity sample_count must be positive")
        stop = first + count
        for start, end, builder in reversed(self._ranges):
            if int(start) <= first and stop <= int(end):
                if callable(builder):
                    return builder(first, count)
                nested = getattr(builder, "build", None)
                if callable(nested):
                    return nested(first, count)
                raise ValueError("identity range has no callable builder")
        raise ValueError("candidate sample lies outside retained identity ranges")

    def __call__(self, first_sample: int, sample_count: int):
        return self.build(first_sample, sample_count)


@dataclass(frozen=True)
class IncoherentDMPlan:
    """Integer channel shifts on one common valid output time axis."""

    dm_trials: np.ndarray
    channel_shifts: np.ndarray
    frequency_hz: np.ndarray
    dump_period_sec: float
    reference_frequency_hz: float
    left_margin: int
    right_margin: int

    @classmethod
    def build(
        cls,
        frequency_hz: Sequence[float],
        dump_period_sec: float,
        dm_trials: Sequence[float],
        reference_frequency_hz: float | None = None,
    ) -> "IncoherentDMPlan":
        frequency = np.asarray(frequency_hz, dtype=np.float64)
        dms = np.asarray(dm_trials, dtype=np.float64)
        if frequency.ndim != 1 or frequency.size == 0:
            raise ValueError("frequency_hz must be a non-empty 1-D array")
        if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
            raise ValueError("frequency_hz must be finite and positive")
        if dms.ndim != 1 or dms.size == 0 or np.any(~np.isfinite(dms)):
            raise ValueError("dm_trials must be a finite non-empty 1-D array")
        if np.unique(dms).size != dms.size:
            raise ValueError("dm_trials must not contain duplicates")
        period = float(dump_period_sec)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("dump_period_sec must be finite and positive")
        reference = (
            float(np.max(frequency))
            if reference_frequency_hz is None
            else float(reference_frequency_hz)
        )
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference_frequency_hz must be finite and positive")
        delay = DISPERSION_DELAY_HZ_SEC * dms[:, None] * (
            frequency[None, :] ** -2 - reference ** -2
        )
        shifts = np.rint(delay / period).astype(np.int64)
        return cls(
            dm_trials=_readonly(dms),
            channel_shifts=_readonly(shifts),
            frequency_hz=_readonly(frequency),
            dump_period_sec=period,
            reference_frequency_hz=reference,
            left_margin=max(0, -int(shifts.min(initial=0))),
            right_margin=max(0, int(shifts.max(initial=0))),
        )

    @property
    def trial_count(self) -> int:
        return int(self.dm_trials.size)

    @property
    def channel_count(self) -> int:
        return int(self.frequency_hz.size)

    def valid_sample_count(self, input_sample_count: int) -> int:
        count = int(input_sample_count) - self.left_margin - self.right_margin
        if count <= 0:
            raise ValueError("DM plan leaves no common valid samples")
        return count


@dataclass(frozen=True)
class StationPowerCalibration:
    """Frozen robust calibration and explicit station/channel weights."""

    location: np.ndarray
    scale: np.ndarray
    station_weights: np.ndarray
    channel_weights: np.ndarray
    background_dumps: int

    def __post_init__(self) -> None:
        location = np.asarray(self.location, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        station = np.asarray(self.station_weights, dtype=np.float64)
        channel = np.asarray(self.channel_weights, dtype=np.float64)
        if location.ndim != 2 or scale.shape != location.shape:
            raise ValueError("location and scale must have [station, channel] shape")
        if station.shape != (location.shape[0],):
            raise ValueError("station_weights do not match calibration")
        if channel.shape != (location.shape[1],):
            raise ValueError("channel_weights do not match calibration")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("calibration scales must be finite and positive")
        if np.any(~np.isfinite(station)) or np.any(station < 0.0):
            raise ValueError("station weights must be finite and non-negative")
        if np.any(~np.isfinite(channel)) or np.any(channel < 0.0):
            raise ValueError("channel weights must be finite and non-negative")
        if not np.any(station > 0.0) or not np.any(channel > 0.0):
            raise ValueError("calibration must retain stations and channels")
        if int(self.background_dumps) <= 0:
            raise ValueError("background_dumps must be positive")
        object.__setattr__(self, "location", _readonly(location))
        object.__setattr__(self, "scale", _readonly(scale))
        object.__setattr__(self, "station_weights", _readonly(station))
        object.__setattr__(self, "channel_weights", _readonly(channel))
        object.__setattr__(self, "background_dumps", int(self.background_dumps))

    @classmethod
    def estimate(
        cls,
        station_power,
        background_mask: Sequence[bool],
        *,
        station_weights=None,
        channel_weights=None,
    ) -> "StationPowerCalibration":
        xp = _array_module(station_power)
        host = xp.asnumpy(station_power) if xp is not np else np.asarray(station_power)
        if host.ndim != 3 or host.size == 0:
            raise ValueError("station_power must be non-empty [time, station, channel]")
        mask = np.asarray(background_mask, dtype=bool)
        if mask.shape != (host.shape[0],) or not np.any(mask):
            raise ValueError("background_mask must select input dumps")
        selected = host[mask]
        location = np.median(selected, axis=0)
        scale = 1.4826 * np.median(np.abs(selected - location[None]), axis=0)
        station = (
            np.ones(host.shape[1], dtype=np.float64)
            if station_weights is None else np.asarray(station_weights, dtype=np.float64)
        )
        channel = (
            np.ones(host.shape[2], dtype=np.float64)
            if channel_weights is None else np.asarray(channel_weights, dtype=np.float64)
        )
        active = (station[:, None] > 0.0) & (channel[None, :] > 0.0)
        if np.any(~np.isfinite(scale[active])) or np.any(scale[active] <= 0.0):
            raise ValueError("background gives invalid active station/channel scale")
        return cls(location, np.where(active, scale, 1.0), station, channel, int(mask.sum()))

    def normalize_and_sum(self, station_power):
        """Return normalized incoherent power as ``[time, channel]`` on-device."""
        if getattr(station_power, "ndim", None) != 3:
            raise ValueError("station_power must have [time, station, channel] shape")
        if tuple(station_power.shape[1:]) != self.location.shape:
            raise ValueError("station_power shape does not match calibration")
        xp = _array_module(station_power)
        dtype = station_power.dtype
        location = xp.asarray(self.location, dtype=dtype)
        scale = xp.asarray(self.scale, dtype=dtype)
        station = xp.asarray(self.station_weights, dtype=dtype)
        channel = xp.asarray(self.channel_weights, dtype=dtype)
        normalized = (station_power - location[None]) / scale[None]
        return (normalized * station[None, :, None]).sum(axis=1) * channel[None]


@dataclass(frozen=True)
class DMSeriesCalibration:
    """Frozen location/scale for each dedispersed DM time series."""

    dm_trials: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    background_samples: int

    def __post_init__(self) -> None:
        dms = np.asarray(self.dm_trials, dtype=np.float64)
        location = np.asarray(self.location, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if dms.ndim != 1 or location.shape != dms.shape or scale.shape != dms.shape:
            raise ValueError("DM calibration arrays must have one common 1-D shape")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("DM calibration scales must be finite and positive")
        if int(self.background_samples) <= 0:
            raise ValueError("background_samples must be positive")
        object.__setattr__(self, "dm_trials", _readonly(dms))
        object.__setattr__(self, "location", _readonly(location))
        object.__setattr__(self, "scale", _readonly(scale))
        object.__setattr__(self, "background_samples", int(self.background_samples))

    @classmethod
    def estimate(cls, dm_series, dm_trials, background_mask):
        xp = _array_module(dm_series)
        host = xp.asnumpy(dm_series) if xp is not np else np.asarray(dm_series)
        dms = np.asarray(dm_trials, dtype=np.float64)
        mask = np.asarray(background_mask, dtype=bool)
        if host.ndim != 2 or dms.shape != (host.shape[0],):
            raise ValueError("dm_series must have [DM, time] shape")
        if mask.shape != (host.shape[1],) or not np.any(mask):
            raise ValueError("background_mask must select DM-series samples")
        selected = host[:, mask]
        location = np.median(selected, axis=1)
        scale = 1.4826 * np.median(np.abs(selected - location[:, None]), axis=1)
        return cls(dms, location, scale, int(mask.sum()))


@dataclass(frozen=True)
class IncoherentTriggerResult:
    """Unthresholded GPU-resident result; candidate policy is a later layer."""

    dm_trials: np.ndarray
    first_input_dump: int
    score: Any
    best_width: Any
    dm_series: Any


def dedisperse_incoherent_power(normalized_power, plan: IncoherentDMPlan):
    """Dedisperse ``[time, channel]`` power without materializing a 3-D cube."""
    if getattr(normalized_power, "ndim", None) != 2:
        raise ValueError("normalized_power must have [time, channel] shape")
    if int(normalized_power.shape[1]) != plan.channel_count:
        raise ValueError("power channel axis does not match DM plan")
    xp = _array_module(normalized_power)
    valid = plan.valid_sample_count(int(normalized_power.shape[0]))
    if xp is not np:
        global _CUPY_DEDISPERSION_KERNEL
        if _CUPY_DEDISPERSION_KERNEL is None:
            _CUPY_DEDISPERSION_KERNEL = xp.RawKernel(
                r"""
                extern "C" __global__
                void dedisperse_power(
                    const float* power,
                    const long long* shifts,
                    float* output,
                    const int trial_count,
                    const int valid_count,
                    const int channel_count,
                    const int left_margin)
                {
                    int index = blockIdx.x;
                    int total = trial_count * valid_count;
                    if (index >= total) return;
                    int dm = index / valid_count;
                    int time = index - dm * valid_count;
                    float sum = 0.0f;
                    int shift_base = dm * channel_count;
                    for (
                        int channel = threadIdx.x;
                        channel < channel_count;
                        channel += blockDim.x
                    ) {
                        long long source_time =
                            (long long)time + left_margin
                            + shifts[shift_base + channel];
                        sum += power[source_time * channel_count + channel];
                    }
                    __shared__ float partial[256];
                    partial[threadIdx.x] = sum;
                    __syncthreads();
                    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                        if (threadIdx.x < offset) {
                            partial[threadIdx.x] += partial[threadIdx.x + offset];
                        }
                        __syncthreads();
                    }
                    if (threadIdx.x == 0) output[index] = partial[0];
                }
                """,
                "dedisperse_power",
            )
        contiguous = xp.ascontiguousarray(normalized_power, dtype=xp.float32)
        shifts = xp.asarray(plan.channel_shifts, dtype=xp.int64)
        output = xp.empty((plan.trial_count, valid), dtype=xp.float32)
        total = plan.trial_count * valid
        _CUPY_DEDISPERSION_KERNEL(
            (total,),
            (256,),
            (
                contiguous, shifts, output, np.int32(plan.trial_count),
                np.int32(valid), np.int32(plan.channel_count),
                np.int32(plan.left_margin),
            ),
        )
        return output
    common = xp.arange(valid, dtype=xp.int64) + plan.left_margin
    channels = xp.arange(plan.channel_count, dtype=xp.int64)
    output = xp.empty((plan.trial_count, valid), dtype=xp.float32)
    for dm_index, shifts in enumerate(plan.channel_shifts):
        indices = common[:, None] + xp.asarray(shifts, dtype=xp.int64)[None]
        output[dm_index] = normalized_power[indices, channels[None]].sum(axis=1)
    return output


def boxcar_bank(
    standardized,
    widths: Sequence[int] = (1, 2, 4, 8, 16, 32),
    *,
    device_widths=None,
):
    """Max-over-width centered boxcars on NumPy or CuPy without host copying."""
    if getattr(standardized, "ndim", None) != 2:
        raise ValueError("standardized must have [DM, time] shape")
    normalized_widths = tuple(sorted(int(width) for width in widths))
    if not normalized_widths or normalized_widths[0] != 1:
        raise ValueError("boxcar widths must be non-empty and include 1")
    if any(width <= 0 for width in normalized_widths):
        raise ValueError("boxcar widths must be positive")
    if len(set(normalized_widths)) != len(normalized_widths):
        raise ValueError("boxcar widths must not contain duplicates")
    xp = _array_module(standardized)
    if xp is not np:
        global _CUPY_BOXCAR_KERNEL
        if _CUPY_BOXCAR_KERNEL is None:
            _CUPY_BOXCAR_KERNEL = xp.RawKernel(
                r"""
                extern "C" __global__
                void boxcar_max(
                    const float* values,
                    const int* widths,
                    float* best,
                    int* best_width,
                    const int dm_count,
                    const int time_count,
                    const int width_count)
                {
                    int index = blockDim.x * blockIdx.x + threadIdx.x;
                    int total = dm_count * time_count;
                    if (index >= total) return;
                    int time = index % time_count;
                    int base = index - time;
                    float maximum = -1.0f / 0.0f;
                    int selected_width = 1;
                    for (int wi = 0; wi < width_count; ++wi) {
                        int width = widths[wi];
                        int start = time - width / 2;
                        if (start < 0 || start + width > time_count) continue;
                        float sum = 0.0f;
                        for (int offset = 0; offset < width; ++offset) {
                            sum += values[base + start + offset];
                        }
                        float score = sum / sqrtf((float)width);
                        if (score > maximum) {
                            maximum = score;
                            selected_width = width;
                        }
                    }
                    best[index] = maximum;
                    best_width[index] = selected_width;
                }
                """,
                "boxcar_max",
            )
        contiguous = xp.ascontiguousarray(standardized, dtype=xp.float32)
        if device_widths is None:
            device_widths = xp.asarray(normalized_widths, dtype=xp.int32)
        best = xp.empty(contiguous.shape, dtype=xp.float32)
        best_width = xp.empty(contiguous.shape, dtype=xp.int32)
        total = int(contiguous.size)
        threads = 128
        _CUPY_BOXCAR_KERNEL(
            ((total + threads - 1) // threads,), (threads,),
            (
                contiguous, device_widths, best, best_width,
                np.int32(contiguous.shape[0]), np.int32(contiguous.shape[1]),
                np.int32(len(normalized_widths)),
            ),
        )
        return best, best_width
    best = xp.full(standardized.shape, -xp.inf, dtype=xp.float32)
    best_width = xp.ones(standardized.shape, dtype=xp.int32)
    for width in normalized_widths:
        if width > standardized.shape[1]:
            continue
        windows = xp.lib.stride_tricks.sliding_window_view(
            standardized, width, axis=1
        )
        # Each local window is reduced in the same order regardless of where
        # the bounded streaming buffer was trimmed.  A global prefix sum would
        # make threshold decisions weakly chunk-boundary dependent.
        summed = windows.sum(axis=-1) / np.sqrt(width)
        center = width // 2
        target = best[:, center : center + summed.shape[1]]
        target_width = best_width[:, center : center + summed.shape[1]]
        improve = summed > target
        target[...] = xp.where(improve, summed, target)
        target_width[...] = xp.where(improve, width, target_width)
    return best, best_width


class IncoherentTriggerKernel:
    """Apply frozen calibration to a contiguous station-power interval."""

    def __init__(
        self,
        plan: IncoherentDMPlan,
        power_calibration: StationPowerCalibration,
        dm_calibration: DMSeriesCalibration,
        *,
        widths: Sequence[int] = (1, 2, 4, 8, 16, 32),
        calibration_version: str = "unversioned",
    ) -> None:
        if not np.array_equal(plan.dm_trials, dm_calibration.dm_trials):
            raise ValueError("DM calibration grid does not match the plan")
        if power_calibration.location.shape[1] != plan.channel_count:
            raise ValueError("power calibration channel count does not match the plan")
        self.plan = plan
        self.power_calibration = power_calibration
        self.dm_calibration = dm_calibration
        self.widths = tuple(widths)
        self.calibration_version = str(calibration_version)

    def update_dm_calibration(
        self, calibration: DMSeriesCalibration, version: str
    ) -> None:
        """Install a version produced only from already-scored past samples."""
        if not np.array_equal(self.plan.dm_trials, calibration.dm_trials):
            raise ValueError("updated DM calibration grid does not match the plan")
        if not version or version == self.calibration_version:
            raise ValueError("updated calibration requires a new non-empty version")
        self.dm_calibration = calibration
        self.calibration_version = str(version)

    def process(self, station_power, *, first_input_dump: int = 0) -> IncoherentTriggerResult:
        normalized = self.power_calibration.normalize_and_sum(station_power)
        return self.process_normalized(normalized, first_input_dump=first_input_dump)

    def process_normalized(
        self, normalized_power, *, first_input_dump: int = 0
    ) -> IncoherentTriggerResult:
        """Search an already normalized/summed contiguous power interval."""
        xp = _array_module(normalized_power)
        series = dedisperse_incoherent_power(normalized_power, self.plan)
        location = xp.asarray(self.dm_calibration.location, dtype=series.dtype)
        scale = xp.asarray(self.dm_calibration.scale, dtype=series.dtype)
        standardized = (series - location[:, None]) / scale[:, None]
        score, best_width = boxcar_bank(standardized, self.widths)
        return IncoherentTriggerResult(
            dm_trials=self.plan.dm_trials,
            first_input_dump=int(first_input_dump) + self.plan.left_margin,
            score=score,
            best_width=best_width,
            dm_series=series,
        )


@dataclass(frozen=True)
class StreamingTriggerBatch:
    """Unique contiguous output owned by one streaming state update."""

    segment_id: int
    first_sample: int
    sample_stride: int
    score: Any
    best_width: Any
    dm_series: Any
    dm_trials: np.ndarray
    identity_builder: Callable[[int, int], Any]
    gap_before: bool
    calibration_version: str = "unversioned"

    @property
    def sample_count(self) -> int:
        return int(self.score.shape[1])

    def sample_id_at(self, index: int):
        if index < 0 or index >= self.sample_count:
            raise IndexError("trigger sample index outside output batch")
        return self.identity_builder(
            self.first_sample + index * self.sample_stride, self.sample_stride
        )


class StreamingIncoherentTrigger:
    """Incremental cross-lease trigger backed by persistent normalized power.

    Input dumps are normalized directly into a device ring.  Each ingest only
    dedisperses the newly eligible output interval plus the narrow boxcar
    guards; previously emitted history is never searched again.
    """

    def __init__(self, kernel: IncoherentTriggerKernel) -> None:
        self.kernel = kernel
        width = max(kernel.widths)
        self._left_boxcar_guard = width // 2
        self._right_boxcar_guard = width - width // 2 - 1
        self._ring = None
        self._ring_capacity = 0
        self._device_constants = None
        self._device_dm_version: str | None = None
        self._retained_from_dump = 0
        self._next_input_dump: int | None = None
        self._next_emit_dump: int | None = None
        self._sample_origin: int | None = None
        self._sample_stride: int | None = None
        self._identity_ranges: list[tuple[int, int, Callable]] = []
        self._segment_id = 0
        self._gap_before_next = False
        self.max_buffered_dumps = 0
        self.dedispersed_samples = 0
        self.max_dedispersed_samples_per_ingest = 0
        self.gap_count = 0
        self.emitted_samples = 0

    @property
    def buffered_dumps(self) -> int:
        if self._ring is None or self._next_input_dump is None:
            return 0
        return int(self._next_input_dump - self._retained_from_dump)

    @property
    def segment_id(self) -> int:
        return self._segment_id

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    def reset(self, *, gap_before: bool = False) -> None:
        # Keep the allocation for the next segment; only logical ownership is
        # reset.  This avoids a device allocation on every discontinuity.
        self._retained_from_dump = 0
        self._next_input_dump = None
        self._next_emit_dump = None
        self._sample_origin = None
        self._sample_stride = None
        self._identity_ranges.clear()
        if gap_before:
            self._segment_id += 1
            self.gap_count += 1
            self._gap_before_next = True

    def _start_segment(self, batch) -> None:
        self._next_input_dump = 0
        self._next_emit_dump = self.kernel.plan.left_margin + self._left_boxcar_guard
        self._sample_origin = int(batch.first_sample)
        self._sample_stride = int(batch.dump_stride_samples)

    def _remember_identity_range(self, batch) -> None:
        start = int(batch.first_sample)
        stop = start + int(batch.dump_count) * int(batch.dump_stride_samples)
        self._identity_ranges.append((start, stop, batch.identity_builder))

    def _identity_builder_snapshot(self):
        return IdentityRangeBuilder(self._identity_ranges)

    def _required_history(self) -> int:
        return (
            self.kernel.plan.left_margin + self.kernel.plan.right_margin
            + self._left_boxcar_guard + self._right_boxcar_guard + 1
        )

    def _ensure_ring(self, xp, incoming: int) -> None:
        required = self._required_history()
        wanted = max(512, required + int(incoming))
        if self._ring is not None and self._ring_capacity >= wanted:
            return
        capacity = 1
        while capacity < wanted:
            capacity *= 2
        replacement = xp.empty(
            (capacity, self.kernel.plan.channel_count), dtype=xp.float32
        )
        if self._ring is not None and self._next_input_dump is not None:
            for logical in range(self._retained_from_dump, self._next_input_dump):
                replacement[logical % capacity] = self._ring[
                    logical % self._ring_capacity
                ]
        self._ring = replacement
        self._ring_capacity = capacity
        if xp is not np and self._device_constants is None:
            calibration = self.kernel.power_calibration
            self._device_constants = {
                "location": xp.asarray(calibration.location, dtype=xp.float32),
                "scale": xp.asarray(calibration.scale, dtype=xp.float32),
                "station": xp.asarray(
                    calibration.station_weights, dtype=xp.float32
                ),
                "channel": xp.asarray(
                    calibration.channel_weights, dtype=xp.float32
                ),
                "shifts": xp.asarray(
                    self.kernel.plan.channel_shifts, dtype=xp.int64
                ),
                "widths": xp.asarray(self.kernel.widths, dtype=xp.int32),
            }

    def _dm_constants(self, xp):
        if xp is np:
            return (
                np.asarray(self.kernel.dm_calibration.location),
                np.asarray(self.kernel.dm_calibration.scale),
            )
        if self._device_dm_version != self.kernel.calibration_version:
            self._device_constants["dm_location"] = xp.asarray(
                self.kernel.dm_calibration.location, dtype=xp.float32
            )
            self._device_constants["dm_scale"] = xp.asarray(
                self.kernel.dm_calibration.scale, dtype=xp.float32
            )
            self._device_dm_version = self.kernel.calibration_version
        return (
            self._device_constants["dm_location"],
            self._device_constants["dm_scale"],
        )

    def _normalize_into_ring(self, station_power, first_dump: int) -> None:
        xp = _array_module(station_power)
        self._ensure_ring(xp, int(station_power.shape[0]))
        if xp is np:
            normalized = self.kernel.power_calibration.normalize_and_sum(station_power)
            for offset in range(int(normalized.shape[0])):
                self._ring[(first_dump + offset) % self._ring_capacity] = normalized[offset]
            return

        global _CUPY_NORMALIZE_RING_KERNEL
        if _CUPY_NORMALIZE_RING_KERNEL is None:
            _CUPY_NORMALIZE_RING_KERNEL = xp.RawKernel(
                r"""
                extern "C" __global__
                void normalize_power_ring(
                    const float* power, const float* location,
                    const float* scale, const float* station_weight,
                    const float* channel_weight, float* ring,
                    const int time_count, const int station_count,
                    const int channel_count, const int ring_capacity,
                    const long long first_dump)
                {
                    int index = blockDim.x * blockIdx.x + threadIdx.x;
                    int total = time_count * channel_count;
                    if (index >= total) return;
                    int time = index / channel_count;
                    int channel = index - time * channel_count;
                    float sum = 0.0f;
                    for (int station = 0; station < station_count; ++station) {
                        int sc = station * channel_count + channel;
                        int input = (time * station_count + station)
                            * channel_count + channel;
                        sum += ((power[input] - location[sc]) / scale[sc])
                            * station_weight[station];
                    }
                    long long slot = (first_dump + time) % ring_capacity;
                    ring[slot * channel_count + channel] =
                        sum * channel_weight[channel];
                }
                """,
                "normalize_power_ring",
            )
        contiguous = xp.ascontiguousarray(station_power, dtype=xp.float32)
        location = self._device_constants["location"]
        scale = self._device_constants["scale"]
        station = self._device_constants["station"]
        channel = self._device_constants["channel"]
        total = int(station_power.shape[0]) * int(station_power.shape[2])
        threads = 256
        _CUPY_NORMALIZE_RING_KERNEL(
            ((total + threads - 1) // threads,), (threads,),
            (
                contiguous, location, scale, station, channel, self._ring,
                np.int32(station_power.shape[0]), np.int32(station_power.shape[1]),
                np.int32(station_power.shape[2]), np.int32(self._ring_capacity),
                np.int64(first_dump),
            ),
        )

    def _dedisperse_ring(self, first_output_dump: int, output_count: int):
        xp = _array_module(self._ring)
        plan = self.kernel.plan
        if xp is np:
            channels = np.arange(plan.channel_count, dtype=np.int64)
            result = np.empty((plan.trial_count, output_count), dtype=np.float32)
            common = np.arange(output_count, dtype=np.int64) + first_output_dump
            for dm_index, shifts in enumerate(plan.channel_shifts):
                source = (common[:, None] + shifts[None]) % self._ring_capacity
                result[dm_index] = self._ring[source, channels[None]].sum(axis=1)
            return result

        global _CUPY_RING_DEDISPERSION_KERNEL
        if _CUPY_RING_DEDISPERSION_KERNEL is None:
            _CUPY_RING_DEDISPERSION_KERNEL = xp.RawKernel(
                r"""
                extern "C" __global__
                void dedisperse_power_ring(
                    const float* ring, const long long* shifts, float* output,
                    const int trial_count, const int output_count,
                    const int channel_count, const int ring_capacity,
                    const long long first_output_dump)
                {
                    int index = blockIdx.x;
                    int total = trial_count * output_count;
                    if (index >= total) return;
                    int dm = index / output_count;
                    int time = index - dm * output_count;
                    float sum = 0.0f;
                    int shift_base = dm * channel_count;
                    for (int channel = threadIdx.x; channel < channel_count;
                         channel += blockDim.x) {
                        long long source = first_output_dump + time
                            + shifts[shift_base + channel];
                        source %= ring_capacity;
                        sum += ring[source * channel_count + channel];
                    }
                    __shared__ float partial[256];
                    partial[threadIdx.x] = sum;
                    __syncthreads();
                    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                        if (threadIdx.x < offset)
                            partial[threadIdx.x] += partial[threadIdx.x + offset];
                        __syncthreads();
                    }
                    if (threadIdx.x == 0) output[index] = partial[0];
                }
                """,
                "dedisperse_power_ring",
            )
        shifts = self._device_constants["shifts"]
        output = xp.empty((plan.trial_count, output_count), dtype=xp.float32)
        _CUPY_RING_DEDISPERSION_KERNEL(
            (plan.trial_count * output_count,), (256,),
            (
                self._ring, shifts, output, np.int32(plan.trial_count),
                np.int32(output_count), np.int32(plan.channel_count),
                np.int32(self._ring_capacity), np.int64(first_output_dump),
            ),
        )
        return output

    def ingest(self, batch) -> StreamingTriggerBatch | None:
        """Consume one ``DetectorSinkBatch`` before its slot lease is released."""
        if int(batch.fft_size) != self.kernel.plan.channel_count:
            raise ValueError("detector batch channel count does not match DM plan")
        if self._sample_stride is not None and int(batch.dump_stride_samples) != self._sample_stride:
            raise RuntimeError("detector dump stride changed inside a segment")
        if self._sample_origin is None:
            self._start_segment(batch)
        else:
            expected_sample = self._sample_origin + int(self._next_input_dump) * self._sample_stride
            if bool(batch.gap_before_first) or int(batch.first_sample) != expected_sample:
                self.reset(gap_before=True)
                self._start_segment(batch)
        self._remember_identity_range(batch)

        first_input = int(self._next_input_dump)
        self._normalize_into_ring(batch.station_power, first_input)
        self._next_input_dump += int(batch.dump_count)
        self.max_buffered_dumps = max(self.max_buffered_dumps, self.buffered_dumps)

        required = self._required_history()
        if self.buffered_dumps < required:
            return None

        eligible_start = self.kernel.plan.left_margin + self._left_boxcar_guard
        eligible_stop = (int(self._next_input_dump) - self.kernel.plan.right_margin
                         - self._right_boxcar_guard)
        emit_start = max(int(self._next_emit_dump), eligible_start)
        if emit_start >= eligible_stop:
            return None
        series_start = emit_start - self._left_boxcar_guard
        series_stop = eligible_stop + self._right_boxcar_guard
        series_count = series_stop - series_start
        series = self._dedisperse_ring(series_start, series_count)
        self.dedispersed_samples += series_count
        self.max_dedispersed_samples_per_ingest = max(
            self.max_dedispersed_samples_per_ingest, series_count
        )
        xp = _array_module(series)
        location, scale = self._dm_constants(xp)
        standardized = (series - location[:, None]) / scale[:, None]
        score, best_width = boxcar_bank(
            standardized, self.kernel.widths,
            device_widths=(
                None if xp is np else self._device_constants["widths"]
            ),
        )
        start = self._left_boxcar_guard
        stop = start + eligible_stop - emit_start
        output = StreamingTriggerBatch(
            segment_id=self._segment_id,
            first_sample=self._sample_origin + emit_start * self._sample_stride,
            sample_stride=self._sample_stride,
            score=score[:, start:stop],
            best_width=best_width[:, start:stop],
            dm_series=series[:, start:stop],
            dm_trials=self.kernel.plan.dm_trials,
            identity_builder=self._identity_builder_snapshot(),
            gap_before=self._gap_before_next,
            calibration_version=self.kernel.calibration_version,
        )
        self._gap_before_next = False
        self._next_emit_dump = eligible_stop
        self.emitted_samples += output.sample_count

        self._retained_from_dump = self._next_emit_dump - (
            self.kernel.plan.left_margin + self._left_boxcar_guard
        )
        retained_sample = (
            self._sample_origin + self._retained_from_dump * self._sample_stride
        )
        self._identity_ranges = [
            item for item in self._identity_ranges if item[1] > retained_sample
        ]
        return output
