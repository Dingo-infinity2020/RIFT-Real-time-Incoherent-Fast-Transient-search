# RIFT v0.1.0 CPU contract

RIFT (Real-time Incoherent Fast Transient search) is released here as a small,
reference **CPU/fake-data contract** for incoherent-search mathematics. It
makes array layout, power integration, normalization, dispersion, and boxcar
scoring reproducible and testable without an instrument or service deployment.

The reference pipeline accepts channelized complex voltage-like arrays, computes
station/channel power (`|V|²`), performs incoherent station summation, applies
normalization and dispersion/boxcar scoring, and returns explicit NumPy results.
The included tests use deterministic fake data and do not contain observation
records.

This release is deliberately not a live acquisition system. It makes no claim
of GPU, RDMA, coherent-beam, hardware, observatory, production, sensitivity, or
real-time performance support. It has no receiver, server, stable-runtime,
operator-control, lifecycle, or network integration, and ships no operational
runbooks, deployment configuration, or observation data.

## Install and test

```text
python -m pip install -e '.[test]'
python -m pytest -q
```

NumPy execution is the only supported public path. Optional array backends in
upstream-derived primitives are outside this release contract and are not
covered by the tests.

## Public modules

The public package contains only the CPU primitives in `src/`:

- `fblock_power.py`: complex-array power integration and gap-safe reduction.
- `incoherent_normalizer.py`: past-only calibration and incoherent summation.
- `incoherent_trigger.py`: dispersion and boxcar scoring over normalized power.

The `tests/` directory contains their focused fake-data tests. No live, network,
hardware, stable-runtime, operator-control, or lifecycle modules are included.

## Provenance

This is a clean, history-free export derived from one upstream source commit.
`public_release_manifest.json` records that commit and SHA-256 for every
exported file except the manifest itself, using repository-relative names only.

## License

Apache License 2.0. See `LICENSE`.
