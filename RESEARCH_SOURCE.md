# Research implementation map

| Module | Role |
|---|---|
| `frontend/` | Component-supervised FMP visual encoder, pixel/object heads and fact extraction |
| `decoder/decoder.py` | Candidate/grid queries and the two-layer semantic object-set decoder |
| `binding/hybrid.py` | Partial same-class association and object-derived fact reconstruction |
| `initial_axis/axis_adapter.py` | Initial native-coordinate guarded screw-axis measurement |
| `readout_v1/experiment.py` | Axis support quality and mask-backed measurement candidate |
| `readout_v2/experiment.py` | Whole-component ownership, completeness and boundary/occlusion gates |
| `binding/original_executor.py` | Exact scalar predicate/normalization functions loaded from the frontend source |
| `explanation/engine.py` | Executed predicate records and compact sufficient alarm certificates |
| `language/` | Predicate/evidence contracts and frozen-language realization interface |

`scripts/` is the verified CPU interface. It consumes frozen facts and scores
instead of starting a neural model or refitting calibration.

Research-source paths have been adapted for distribution. The optional environment
variables `SGDB_ARCHIVE_ROOT`, `SGDB_FRONTEND_ROOT`, `SGDB_INITIAL_AXIS_ROOT` and
`SGDB_IMAGENET_WEIGHTS` point to separately supplied research assets. Original
training/evaluation phase controllers require those historical assets; do not run
them to reproduce the supplied tables. No server launcher, login helper or
credential is included. The CPU verification does not import training libraries.

The original Ropes OCR witness uses frozen Qwen2.5-VL-3B transcription. The final
explanation model uses frozen Qwen3-VL-8B in text-only mode. `language/qwen_adapter.py` provides the optional frozen text-only generation adapter
with the original greedy decoding settings. It is not invoked during CPU reproduction.
Recorded OCR facts and
the final language outputs are supplied, while the external model binaries remain
with their providers.
