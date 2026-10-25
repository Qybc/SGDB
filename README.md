# SGDB

**Semantic-Geometric Decoupling and Evidence Binding for Logical Anomaly Detection**

Research implementation and frozen-result reproduction materials accompanying
the SGDB manuscript. The released detector configuration is **complete_support V2**.

SGDB decouples learned object identity from measured geometry, binds geometric
witnesses to recognized objects, and checks instance support before replacing
measurements. Executed constraints produce decisions and source-linked evidence;
compact score-sufficient certificates supply the predicates for frozen-LLM reports.

## Results

| Dataset / scope | AUROC (%) | Fixed-threshold F1 (%) |
|---|---:|---:|
| MVTec LOCO AD, five logical-anomaly tasks | 98.19 | 94.52 |
| VID-AD, all 50 deployment tasks | 93.56 | 92.76 |

All numbers are task macro-averages. See [comparison conditions](comparison_conditions.csv)
for supervision, normal-data, calibration and implementation settings. Internal
mechanism controls use the shared component supervision and task requirements.

## Quick start: CPU reproduction

Tested with Python 3.12.14 and NumPy 2.3.5. No dataset download, GPU, neural-model
call, training, baseline inference or threshold fitting is needed for these commands.

```sh
git clone https://github.com/Qybc/SGDB.git
cd SGDB
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-cpu.txt
python scripts/verify_integrity.py
python scripts/reproduce.py --out reproduction_output
python scripts/replay_facts.py --out fact_replay.json
python scripts/replay_language.py --out language_replay.json
```

Choose a new output path when rerunning. Archived `results/` are preserved.

| Entry point | Input | Verified output |
|---|---|---|
| `scripts/reproduce.py` | Saved per-image scores, labels, groups, thresholds | 990 method/task rows; 715 manuscript rows checked; paired intervals and transition ledgers |
| `scripts/replay_facts.py` | Frozen objects/facts, references and rules | 9,031 scores/decisions; 66,297 executed rules; explanation certificates |
| `scripts/replay_language.py` | 70 saved language responses and contracts | Evidence-contract replay with zero new LLM calls |
| `scripts/verify_bootstrap.py` | Saved group draws and bootstrap metrics | 144 independent expanded-sample/pairwise-AUROC checks |

To inspect one source-linked explanation:

```sh
python scripts/replay_facts.py \
  --source-id Tapes/test/logical_anomalies/Length/000.jpg \
  --out tape_trace.json
```

The structured replay does not generate new language-model text. The separate
language replay validates the recorded frozen-LLM outputs. Mechanical contract
validity and score sufficiency have their stated scope.

## Repository contents

- `research_source/`: visual frontend, semantic decoder, partial binding, geometric
  readout, executor, trace explanation and language-contract implementations.
- `data/`: 162,558 frozen predictions for 18 configurations, 12,666 final fact
  records, role/group metadata and normal component targets.
- `configs/`: frozen normal intervals, residual references, task thresholds and
  development-selected settings. `SGDB_V2` is final; `initial_bound` is its initial-readout control.
- `results/`: task/category/condition metrics, 10,000 paired grouped-bootstrap
  draws, marginal 95% intervals, and repaired/regressed/unchanged group ledgers.
- `language_records/`: 70 frozen outputs with their evidence contracts.
- `checkpoints/semantic_decoder/`: 15 already trained final semantic-decoder heads.
- `audit/`: numerical checks, group checks, model-asset hashes and runtime metadata.

## Research modules and original-image inference

The CPU entry points above reproduce the released analyses from frozen outputs
and facts. The research modules document the original learned frontend/decoder,
binding and readout algorithms. Their full historical training controllers are
not the CPU quick-start interface. See [research-source notes](RESEARCH_SOURCE.md).

The 15 visual-frontend weights and pretrained OCR/LLM runtime assets are not
included in this snapshot; the checkpoint hashes are listed in
[`audit/model_inventory.json`](audit/model_inventory.json). Raw benchmark images
must be obtained from their original providers. A new [raw-image inference example](RAW_IMAGE_DEMO.md)
connects the frozen models, V2 readout, executor and Qwen explanation backend for
`screw_bag`. Two original-image runs matched the archived scores and decisions
exactly. The visual-frontend checkpoint and local Qwen model are separate assets.
Other categories retain the frozen-output interface. `scripts/check_assets.py`
checks optional supplied checkpoints without running a model.

## Evaluation protocol

The 10,000 bootstrap replicates (seed 20261025) keep all methods paired and sample
within category/label strata. VID holds the five capture conditions together by
scene, anomaly-path type and filename index (1,579 groups). These index groups do
not establish physical-instance or source-video independence. Public test cases
informed earlier development; the reported intervals describe conditional cohort
stability, rather than a new blind test or training-seed variation. Each contrast
has a marginal interval, without simultaneous multiple-comparison adjustment.

The final readout preserves LOCO performance and retains tape corrections on VID.
The uniform-mask control has a small VID F1 advantage; the complete-support
mechanism is evaluated for its cross-category measurement behavior.

## Data sources and citation

MVTec LOCO AD: https://www.mvtec.com/company/research/datasets/mvtec-loco

VID-AD: use the provider cited in the manuscript (Video-Induced Distribution
Shift for Logical Anomaly Detection in Industrial Inspection,
https://doi.org/10.48550/arXiv.2603.13964).

The benchmark images and third-party pretrained model binaries are not republished
here. Normal component targets, frozen measurements and outputs are supplied for
the documented research protocol. Citation metadata are in [`CITATION.cff`](CITATION.cff).
Use the fixed **v1.0.0** release for the manuscript's numerical reproduction.
