# Third-party sources

## ERNIE compatibility sources

`vendor/ernie/` contains Baidu's ERNIE configuration, tokenizer and modeling code used by this experiment. Original copyright headers and the upstream Apache-2.0 license are retained in `vendor/ernie/LICENSE`.

The modeling file includes the local vectorized dispatch compatibility implementation used in the recorded runs. The overlay also pins the configuration and tokenizer mappings. The preparation script installs these files into a newly downloaded checkpoint and preserves the upstream copies; it does not change model weights.

Model weights are not redistributed. The base checkpoint is `baidu/ERNIE-4.5-21B-A3B-PT`, revision `23c6cb79af127ec966c9c03506dc80a3ff9fa953`.

## Granite

Granite weights and tokenizer files are downloaded from `ibm-granite/granite-3.1-1b-a400m-base`, revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`. They are not included in this repository and retain their original license.

## Datasets and evaluation

ARC-Easy and OpenBookQA are downloaded through lm-evaluation-harness. Their original terms apply. This repository includes row indices, hashes and derived scores, not question text or a redistributed dataset.

The project uses NumPy, PyYAML, PyTorch, Transformers, Accelerate, Hugging Face Datasets and lm-evaluation-harness under their respective licenses. No installed dependencies or model weight caches are vendored.

## Report template

The report source references the NeurIPS 2026 style file. Obtain it from the official template location listed in the `.tex` header; the style file is not included here.
