# SPDX-License-Identifier: Apache-2.0
"""Pure-NumPy robust calibration and incoherent power normalization.

This module consumes the M1.1 :class:`PowerIntegrationResult` contract.  Its
``valid_weight`` is the static ``float64[512]`` count of valid fine channels
per reduced channel, so this implementation broadcasts it as
``valid_weight[None, None, :]`` over ``[time, stream, channel]`` power.

Calibration is a frozen, past-only snapshot.  It stores per-stream/channel
median location and ``1.4826 * MAD`` scale, explicit boolean stream and
channel masks, and provenance for the half-open history range and its
validity horizon.  Applying a missing, stale, mismatched, or not-ready
snapshot returns a finite detector-disabled result; it never mutates or
blocks the upstream power input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

import numpy as np

from .fblock_power import INTEGRATION_SAMPLES


LOGICAL_STREAMS = 8
OUTPUT_CHANNELS = 512
MAD_SCALE_FACTOR = 1.4826


class NormalizationStatus(str, Enum):
    """Detector-side status for one normalization attempt."""

    ENABLED = "ENABLED"
    DISABLED = "DISABLED"


class _InvalidInput(ValueError):
    """Internal validation error converted to a disabled detector result."""


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_hash(value: Any, name: str) -> str:
    text = _strict_text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return text


def _mask(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be converted to an array") from exc
    if array.shape != shape or array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have shape {shape} and boolean dtype")
    return np.array(array, dtype=bool, copy=True)


def _power(value: Any, *, name: str, allow_empty: bool) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise _InvalidInput(f"{name} cannot be converted to an array") from exc
    if array.ndim != 3 or array.shape[1:] != (LOGICAL_STREAMS, OUTPUT_CHANNELS):
        raise _InvalidInput(f"{name} must have shape (time, 8, 512)")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError(f"{name} history must be non-empty")
    if array.dtype != np.dtype(np.float64):
        raise _InvalidInput(f"{name} must have exact float64 dtype")
    if not np.all(np.isfinite(array)):
        raise _InvalidInput(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise _InvalidInput(f"{name} must contain non-negative power")
    return array


@dataclass(frozen=True)
class CalibrationProvenance:
    """Immutable identity and validity metadata for a calibration snapshot."""

    history_first_sample_index: int
    history_stop_sample_exclusive: int
    layout_hash: str
    frequency_hash: str
    detector_config_hash: str
    input_identifier: str
    valid_until_sample_exclusive: int

    def __post_init__(self) -> None:
        first = _strict_nonnegative_int(
            self.history_first_sample_index, "history_first_sample_index"
        )
        stop = _strict_nonnegative_int(
            self.history_stop_sample_exclusive, "history_stop_sample_exclusive"
        )
        valid_until = _strict_nonnegative_int(
            self.valid_until_sample_exclusive, "valid_until_sample_exclusive"
        )
        if stop <= first:
            raise ValueError("history range must be non-empty and half-open")
        if valid_until < stop:
            raise ValueError("calibration validity cannot end before history stop")
        values = {
            "layout_hash": _strict_hash(self.layout_hash, "layout_hash"),
            "frequency_hash": _strict_hash(self.frequency_hash, "frequency_hash"),
            "detector_config_hash": _strict_hash(
                self.detector_config_hash, "detector_config_hash"
            ),
            "input_identifier": _strict_text(self.input_identifier, "input_identifier"),
        }
        object.__setattr__(self, "history_first_sample_index", first)
        object.__setattr__(self, "history_stop_sample_exclusive", stop)
        object.__setattr__(self, "valid_until_sample_exclusive", valid_until)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def history_range(self) -> tuple[int, int]:
        """Half-open ``[first, stop)`` ADC-sample history range."""

        return (
            self.history_first_sample_index,
            self.history_stop_sample_exclusive,
        )


@dataclass(frozen=True)
class RobustCalibration:
    """Frozen median/MAD calibration for eight logical streams and 512 channels.

    A calibration with ``ready=False`` is an explicit not-ready snapshot.  It
    is safe to retain and audit, but :class:`IncoherentNormalizer` will return
    ``DISABLED`` rather than applying it.  For a ready snapshot every active
    stream/channel scale is finite and strictly positive.  Inactive entries
    are ignored by application regardless of their stored values.
    """

    location: np.ndarray
    scale: np.ndarray
    stream_mask: np.ndarray
    channel_mask: np.ndarray
    provenance: CalibrationProvenance
    ready: bool = True
    not_ready_reason: str | None = None

    def __post_init__(self) -> None:
        location = np.asarray(self.location)
        scale = np.asarray(self.scale)
        if location.shape != (LOGICAL_STREAMS, OUTPUT_CHANNELS) or scale.shape != (
            LOGICAL_STREAMS,
            OUTPUT_CHANNELS,
        ):
            raise ValueError("location and scale must have shape (8, 512)")
        if location.dtype != np.dtype(np.float64) or scale.dtype != np.dtype(np.float64):
            raise ValueError("location and scale must have exact float64 dtype")
        if not np.all(np.isfinite(location)) or not np.all(np.isfinite(scale)):
            raise ValueError("location and scale must be finite")
        if np.any(scale < 0.0):
            raise ValueError("scale must be non-negative")
        stream_mask = _mask(self.stream_mask, (LOGICAL_STREAMS,), "stream_mask")
        channel_mask = _mask(self.channel_mask, (OUTPUT_CHANNELS,), "channel_mask")
        if not isinstance(self.provenance, CalibrationProvenance):
            raise ValueError("provenance must be CalibrationProvenance")
        if type(self.ready) is not bool:
            raise ValueError("ready must be boolean")
        active = stream_mask[:, None] & channel_mask[None, :]
        scales_ready = np.all(((scale > 0.0) & np.isfinite(scale)) | ~active)
        ready = bool(self.ready)
        reason = self.not_ready_reason
        if ready and not np.any(active):
            raise ValueError("ready calibration must have an active stream and channel")
        if ready and not scales_ready:
            raise ValueError("ready calibration has nonpositive active scale")
        if not ready:
            if reason is None:
                reason = "calibration_not_ready"
            reason = _strict_text(reason, "not_ready_reason")
        elif reason is not None:
            raise ValueError("ready calibration cannot carry not_ready_reason")
        location = _readonly(location)
        scale = _readonly(scale)
        stream_mask = _readonly(stream_mask)
        channel_mask = _readonly(channel_mask)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "stream_mask", stream_mask)
        object.__setattr__(self, "channel_mask", channel_mask)
        object.__setattr__(self, "not_ready_reason", reason)

    @classmethod
    def fit(
        cls,
        history_power: Any,
        *,
        history_first_sample_index: Any,
        history_stop_sample_exclusive: Any,
        valid_until_sample_exclusive: Any,
        layout_hash: Any,
        frequency_hash: Any,
        detector_config_hash: Any,
        input_identifier: Any,
        stream_mask: Any,
        channel_mask: Any,
    ) -> "RobustCalibration":
        """Fit an immutable median/MAD snapshot from non-empty past power only."""

        history = _power(history_power, name="history_power", allow_empty=False)
        stream = _mask(stream_mask, (LOGICAL_STREAMS,), "stream_mask")
        channel = _mask(channel_mask, (OUTPUT_CHANNELS,), "channel_mask")
        provenance = CalibrationProvenance(
            history_first_sample_index=_strict_nonnegative_int(
                history_first_sample_index, "history_first_sample_index"
            ),
            history_stop_sample_exclusive=_strict_nonnegative_int(
                history_stop_sample_exclusive, "history_stop_sample_exclusive"
            ),
            layout_hash=layout_hash,
            frequency_hash=frequency_hash,
            detector_config_hash=detector_config_hash,
            input_identifier=input_identifier,
            valid_until_sample_exclusive=_strict_nonnegative_int(
                valid_until_sample_exclusive, "valid_until_sample_exclusive"
            ),
        )
        expected_history_samples = int(history.shape[0]) * INTEGRATION_SAMPLES
        if (
            provenance.history_stop_sample_exclusive
            - provenance.history_first_sample_index
            != expected_history_samples
        ):
            raise ValueError(
                "history range must equal history_power rows times INTEGRATION_SAMPLES"
            )
        location = np.asarray(np.median(history, axis=0), dtype=np.float64)
        deviations = np.abs(history - location[None, :, :])
        scale = np.asarray(MAD_SCALE_FACTOR * np.median(deviations, axis=0), dtype=np.float64)
        active = stream[:, None] & channel[None, :]
        ready_scales = np.isfinite(scale) & (scale > 0.0)
        ready = bool(np.any(active) and np.all(ready_scales | ~active))
        if ready:
            reason = None
        elif not np.any(active):
            reason = "no_active_stream_or_channel"
        else:
            reason = "nonpositive_active_scale"
        return cls(
            location=location,
            scale=scale,
            stream_mask=stream,
            channel_mask=channel,
            provenance=provenance,
            ready=ready,
            not_ready_reason=reason,
        )

    from_history = fit


def fit_robust_calibration(history_power: Any, **kwargs: Any) -> RobustCalibration:
    """Functional alias for :meth:`RobustCalibration.fit`."""

    return RobustCalibration.fit(history_power, **kwargs)


def _metadata(
    first: Any,
    stop: Any,
    gap: Any,
    n_time: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        first_array = np.asarray(first)
        stop_array = np.asarray(stop)
        gap_array = np.asarray(gap)
    except (TypeError, ValueError) as exc:
        raise _InvalidInput("sample metadata cannot be converted to arrays") from exc
    if (
        first_array.shape != (n_time,)
        or stop_array.shape != (n_time,)
        or first_array.dtype.kind not in "iu"
        or stop_array.dtype.kind not in "iu"
    ):
        raise _InvalidInput("sample metadata must contain integer arrays matching time")
    if gap_array.shape != (n_time,) or gap_array.dtype != np.dtype(bool):
        raise _InvalidInput("gap_before must be a boolean array matching time")
    if np.any(first_array < 0) or np.any(stop_array <= first_array):
        raise _InvalidInput("sample metadata must define positive half-open ranges")
    expected_stop = first_array + INTEGRATION_SAMPLES
    if np.any(stop_array != expected_stop):
        raise _InvalidInput("every science range must span one integration")
    for index in range(1, n_time):
        previous_stop = stop_array[index - 1]
        current_first = first_array[index]
        if current_first < previous_stop:
            raise _InvalidInput("science sample ranges overlap or move backward")
        if gap_array[index]:
            continue
        if current_first != previous_stop:
            raise _InvalidInput("undeclared science sample hole")
    return (
        np.array(first_array, copy=True),
        np.array(stop_array, copy=True),
        np.array(gap_array, copy=True),
    )


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class IncoherentNormalizationResult:
    """Normalized detector output or an explicit finite disabled result."""

    science: np.ndarray
    normalized_residual: np.ndarray
    monitor: np.ndarray
    channel_valid: np.ndarray
    first_frame_sample_index: np.ndarray
    stop_sample_exclusive: np.ndarray
    gap_before: np.ndarray
    status: NormalizationStatus
    disabled_reason: str | None = None

    def __post_init__(self) -> None:
        science = np.asarray(self.science)
        residual = np.asarray(self.normalized_residual)
        monitor = np.asarray(self.monitor)
        channel_valid = np.asarray(self.channel_valid)
        first = np.asarray(self.first_frame_sample_index)
        stop = np.asarray(self.stop_sample_exclusive)
        gap = np.asarray(self.gap_before)
        if science.ndim != 2 or science.shape[1:] != (OUTPUT_CHANNELS,):
            raise ValueError("science must have shape (time, 512)")
        n_time = science.shape[0]
        if residual.shape != (n_time, LOGICAL_STREAMS, OUTPUT_CHANNELS):
            raise ValueError("normalized_residual must have shape (time, 8, 512)")
        if monitor.shape != (n_time, LOGICAL_STREAMS):
            raise ValueError("monitor must have shape (time, 8)")
        if channel_valid.shape != (OUTPUT_CHANNELS,) or channel_valid.dtype != np.dtype(bool):
            raise ValueError("channel_valid must have shape (512,) and boolean dtype")
        if first.shape != (n_time,) or stop.shape != (n_time,):
            raise ValueError("sample ranges must match science time")
        if first.dtype.kind not in "iu" or stop.dtype.kind not in "iu":
            raise ValueError("sample ranges must have integer dtype")
        if gap.shape != (n_time,) or gap.dtype != np.dtype(bool):
            raise ValueError("gap_before must have shape (time,) and boolean dtype")
        for array in (science, residual, monitor):
            if array.dtype != np.dtype(np.float64) or not np.all(np.isfinite(array)):
                raise ValueError("normalized outputs must be finite float64 arrays")
        status = NormalizationStatus(self.status)
        reason = self.disabled_reason
        if status is NormalizationStatus.DISABLED:
            if reason is None:
                raise ValueError("disabled result must include disabled_reason")
            reason = _strict_text(reason, "disabled_reason")
        elif reason is not None:
            raise ValueError("enabled result cannot include disabled_reason")
        object.__setattr__(self, "science", _readonly(science))
        object.__setattr__(self, "normalized_residual", _readonly(residual))
        object.__setattr__(self, "monitor", _readonly(monitor))
        object.__setattr__(self, "channel_valid", _readonly(channel_valid))
        object.__setattr__(self, "first_frame_sample_index", _readonly(first))
        object.__setattr__(self, "stop_sample_exclusive", _readonly(stop))
        object.__setattr__(self, "gap_before", _readonly(gap))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "disabled_reason", reason)

    @property
    def enabled(self) -> bool:
        return self.status is NormalizationStatus.ENABLED

    @property
    def science_power(self) -> np.ndarray:
        return self.science

    @property
    def residual(self) -> np.ndarray:
        return self.normalized_residual

    @property
    def per_stream_monitor(self) -> np.ndarray:
        return self.monitor


class IncoherentNormalizer:
    """Apply a frozen robust calibration without blocking upstream power."""

    def __init__(self, calibration: RobustCalibration | None):
        if calibration is not None and not isinstance(calibration, RobustCalibration):
            raise ValueError("calibration must be RobustCalibration or None")
        self.calibration = calibration

    @classmethod
    def from_history(cls, history_power: Any, **kwargs: Any) -> "IncoherentNormalizer":
        return cls(RobustCalibration.fit(history_power, **kwargs))

    @staticmethod
    def _disabled(
        n_time: int,
        first: np.ndarray | None,
        stop: np.ndarray | None,
        gap: np.ndarray | None,
        reason: str,
    ) -> IncoherentNormalizationResult:
        if first is None or stop is None or gap is None:
            first = np.empty((n_time,), dtype=np.int64)
            stop = np.empty((n_time,), dtype=np.int64)
            gap = np.zeros((n_time,), dtype=bool)
        return IncoherentNormalizationResult(
            science=np.zeros((n_time, OUTPUT_CHANNELS), dtype=np.float64),
            normalized_residual=np.zeros(
                (n_time, LOGICAL_STREAMS, OUTPUT_CHANNELS), dtype=np.float64
            ),
            monitor=np.zeros((n_time, LOGICAL_STREAMS), dtype=np.float64),
            channel_valid=np.zeros((OUTPUT_CHANNELS,), dtype=bool),
            first_frame_sample_index=first,
            stop_sample_exclusive=stop,
            gap_before=gap,
            status=NormalizationStatus.DISABLED,
            disabled_reason=reason,
        )

    def apply(
        self,
        power_result: Any,
        *,
        layout_hash: Any = None,
        frequency_hash: Any = None,
        detector_config_hash: Any = None,
        input_identifier: Any = None,
    ) -> IncoherentNormalizationResult:
        """Normalize one M1.1 result, returning ``DISABLED`` on detector errors."""

        n_time = 0
        first: np.ndarray | None = None
        stop: np.ndarray | None = None
        gap: np.ndarray | None = None
        try:
            power = _power(power_result.power, name="science power", allow_empty=True)
            n_time = int(power.shape[0])
            first, stop, gap = _metadata(
                power_result.first_frame_sample_index,
                power_result.stop_sample_exclusive,
                power_result.gap_before,
                n_time,
            )
            weight = np.asarray(power_result.valid_weight)
            if weight.shape != (OUTPUT_CHANNELS,) or weight.dtype != np.dtype(np.float64):
                raise _InvalidInput("valid_weight must have shape (512,) and float64 dtype")
            if (
                not np.all(np.isfinite(weight))
                or np.any(weight < 0.0)
                or np.any(weight > 8.0)
                or np.any(weight != np.floor(weight))
            ):
                raise _InvalidInput("valid_weight must be finite integer counts within [0, 8]")
        except (AttributeError, TypeError, ValueError, _InvalidInput) as exc:
            if n_time == 0 and power_result is not None:
                try:
                    candidate = np.asarray(power_result.power)
                    if candidate.ndim >= 1:
                        n_time = int(candidate.shape[0])
                except (AttributeError, TypeError, ValueError):
                    pass
            return self._disabled(n_time, first, stop, gap, f"invalid_input:{exc}")

        calibration = self.calibration
        if calibration is None:
            return self._disabled(n_time, first, stop, gap, "missing_calibration")
        if not calibration.ready:
            return self._disabled(
                n_time,
                first,
                stop,
                gap,
                calibration.not_ready_reason or "calibration_not_ready",
            )
        try:
            runtime_values = (
                _strict_hash(layout_hash, "layout_hash"),
                _strict_hash(frequency_hash, "frequency_hash"),
                _strict_hash(detector_config_hash, "detector_config_hash"),
                _strict_text(input_identifier, "input_identifier"),
            )
        except ValueError as exc:
            return self._disabled(n_time, first, stop, gap, f"missing_provenance:{exc}")
        expected_values = (
            calibration.provenance.layout_hash,
            calibration.provenance.frequency_hash,
            calibration.provenance.detector_config_hash,
            calibration.provenance.input_identifier,
        )
        if runtime_values != expected_values:
            return self._disabled(n_time, first, stop, gap, "provenance_mismatch")
        if n_time:
            if calibration.provenance.history_stop_sample_exclusive > int(first[0]):
                return self._disabled(n_time, first, stop, gap, "history_science_overlap")
            if int(stop[-1]) > calibration.provenance.valid_until_sample_exclusive:
                return self._disabled(n_time, first, stop, gap, "calibration_expired")

        active_channels = (
            calibration.channel_mask & (weight > 0.0)
        )
        if not np.any(active_channels):
            return self._disabled(n_time, first, stop, gap, "no_effective_bandwidth")
        active = calibration.stream_mask[:, None] & active_channels[None, :]
        channel_count = active.sum(axis=0).astype(np.float64)
        residual = np.zeros_like(power, dtype=np.float64)
        np.subtract(
            power,
            calibration.location[None, :, :],
            out=residual,
            where=np.ones_like(power, dtype=bool),
        )
        np.divide(
            residual,
            calibration.scale[None, :, :],
            out=residual,
            where=active[None, :, :],
        )
        residual *= active[None, :, :]
        channel_valid = channel_count > 0.0
        science = np.zeros((n_time, OUTPUT_CHANNELS), dtype=np.float64)
        if np.any(channel_valid):
            summed = residual.sum(axis=1, dtype=np.float64)
            science[:, channel_valid] = summed[:, channel_valid] / np.sqrt(
                channel_count[channel_valid][None, :]
            )
        monitor = np.zeros((n_time, LOGICAL_STREAMS), dtype=np.float64)
        if np.any(active_channels):
            for stream_index in np.flatnonzero(calibration.stream_mask):
                monitor[:, stream_index] = residual[:, stream_index, active_channels].mean(axis=1)
        return IncoherentNormalizationResult(
            science=science,
            normalized_residual=residual,
            monitor=monitor,
            channel_valid=channel_valid,
            first_frame_sample_index=first,
            stop_sample_exclusive=stop,
            gap_before=gap,
            status=NormalizationStatus.ENABLED,
        )

    process = apply
    normalize = apply


RobustPowerCalibration = RobustCalibration
IncoherentNormalization = IncoherentNormalizationResult
PowerCalibration = RobustCalibration
NormalizedPowerResult = IncoherentNormalizationResult
fit_calibration = fit_robust_calibration
