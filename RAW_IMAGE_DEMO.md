# Raw image inference example

This entry point runs an RGB image through the frozen visual frontend, semantic
fact decoder, geometric witness construction, binding, complete_support V2 readout,
constraint executor, execution certificate and frozen Qwen explanation backend.
It is currently validated for the **screw_bag** category. It does not train models,
fit thresholds or launch baseline inference.

## Assets and environment

- Obtain the benchmark images from the MVTec LOCO AD provider.
- Supply the existing screw_bag visual-frontend checkpoint. SHA256:
  `f5cef4c8a152e4b1c5925d74b38d65cb4e2ec591e6540f476f5c68d555507172`.
  This separate checkpoint is 126,428,230 bytes; it is outside the Git source snapshot.
- The semantic decoder is included at
  `checkpoints/semantic_decoder/screw_bag/best.pt` (SHA256
  `d6cdd994c39d95724640e202ba5c0c854a3e0b1135daef84f55f066f81af9022`).
- Supply a local `Qwen3-VL-8B-Instruct` model directory for natural-language output.
  The model is frozen and used with text input. It does not inspect the image or
  change the detector decision. No model is downloaded automatically.
- The verified runtime used Python 3.10.12, PyTorch 2.5.1 with CUDA 12.4,
  torchvision 0.20.1, NumPy 1.26.4, SciPy 1.15.3, Pillow 12.2.0,
  scikit-image 0.25.2, Transformers 5.12.1 and OpenCV. See
  `audit/research_runtime.json` for the recorded base-image metadata.

Install the research dependencies in a compatible CUDA environment using
`requirements-research.txt`. The CPU reproduction environment is separate.

## Run

Expose only the intended physical GPU. Both verified runs used physical GPU 2.

```sh
CUDA_VISIBLE_DEVICES=2 python scripts/infer_image.py \
  --image /path/to/image.png \
  --frontend /path/to/screw_bag_frontend.pt \
  --llm-model /path/to/Qwen3-VL-8B-Instruct \
  --out raw_demo_output
```

For Chinese explanations, add `--language zh`. Choose a new output directory for
each run. Omitting `--llm-model` produces the detection and structured evidence
without running the language model.

The input path is only used to load pixels. The model receives no test annotation
or expected decision. The inference command does not read archived per-image
facts, predictions, language outputs or test labels. Normal-fitted deployment
constants and task constraints are loaded from the frozen configuration.

The scalar executor runs in a separate CPU process via `scripts/trace_image.py`.
The language backend receives only the selected executed predicates. The final
JSON includes the unchanged numeric evidence and contract-validation result.

## Outputs

| File | Contents |
|---|---|
| `result.json` | Decision, score, threshold, generated explanation and evidence-file links |
| `execution_trace.json` | Executed rules, operands, provenance and sufficient-rule certificate |
| `language.json` | Fresh Qwen output, source-bound evidence and validation result |
| `evidence_overlay.png` | Measured instances and axes on the input image |
| `facts.json`, `stages.json` | Fresh final facts and intermediate recognition/binding results |
| `support_events.json`, `screw_support.png` | Support-aware readout diagnostics and foreground support |
| `run_receipt.json` | Input/model/configuration hashes, environment and measured run times |

## Verified examples

Two raw images were run on 25 October 2026. Archived outputs were accessed only
after both runs, in a separate regression comparison.

| Source image | Fresh score | Fixed threshold | Decision | Score difference from archive |
|---|---:|---:|---|---:|
| `screw_bag/test/good/000.png` | 0.000000 | 4.430046 | No alarm | 0 |
| `screw_bag/test/logical_anomalies/000.png` | 6.130061 | 4.430046 | Anomaly | 0 |

For the anomaly example, the short-screw measurement is 428.2202 px and its normal
upper bound is 363.1222 px. The generated finding is
“The screw length exceeds the upper bound. [E05]”. The evidence table identifies
the measurement, interval, object index and source fact path.

Both fresh language outputs passed the contract checks. Object and final fact
records matched the frozen archive exactly; support diagnostics differed by at
most 2.3e-13. `audit/raw_image_verification.json` records the comparison. These two
samples verify the inference implementation and do not constitute a new benchmark
or blind test. Other categories retain the frozen-output reproduction interface
and have not been certified through this new raw-image entry point.

The original v1.0.0 tag remains the fixed numerical-reproduction snapshot.
