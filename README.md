# Semantic Predictive Coding Pipeline

This repository contains the current working semantic-only pipeline for the
Twin Mansion predictive coding experiments. Large data files, model checkpoints,
notebooks, old multimodal scripts, and generated analysis outputs are intentionally
excluded from version control.

## Current Focus

The active code path studies representations learned from semantic object-label
sequences, without visual frames or position supervision.

The core workflow is:

1. Train a pure semantic Transformer model with `train_semantic_gpt.py`.
2. Train a frame-wise semantic autoencoder baseline with `train_semantic_ae.py`.
3. Export cached latents on predefined trajectories with `semantic_gpt_latent.py`
   or the extraction step in `train_semantic_ae.py`.
4. Analyze latent structure with position decoding, vocabulary selectivity, and
   placefield-style spatial projections.

## Data Layout

Data is not included in the repository. The scripts expect local files with this
general structure:

```text
data_root/
  vocabulary.npy
  data_...samples/
    path1/
      objects_map.npy
      positions.npy
    path2/
      objects_map.npy
      positions.npy
  pre_defined_path_samples/
    objects_map.npy
    positions.npy
```

`objects_map.npy` is a frame-by-vocabulary multi-hot matrix. `vocabulary.npy`
contains the object names whose indices match the columns of `objects_map.npy`.
Object names are split on underscores to form subword token sequences.

## Models

`SemanticGPT` is defined in `src/models/semantic_gpt.py`.

- Input: token ids and masks with shape `(B, L, K)`.
- Default sequence length: `L=25`.
- Default tokens per frame: `K=16`.
- Object names are tokenized into subwords, embedded, and given frame-only
  sinusoidal positional encodings.
- Frame-token pairs are flattened to length `L*K` and processed by causal
  self-attention blocks.
- The final per-frame latent is a masked mean over tokens, shape `(B, L, D)`.
- The head predicts a multi-label object vector for each frame.

`SemanticFrameAutoencoder` is defined in `src/models/semantic_autoencoder.py`.

- It flattens time into the batch dimension: `(B, L, K) -> (B*L, K)`.
- It reconstructs each frame from only that frame's semantic tokens.
- It does not receive other frames, position, images, path ids, or temporal order.
- It is used as a reconstruction baseline for the semantic latent analyses.

Important caveat: the current `train_semantic_gpt.py` objective predicts the
same-frame semantic multi-hot target, not a shifted next-frame target. This makes
the current Transformer closer to a causal semantic reconstruction model than a
strict next-step predictive coding model.

## Main Scripts

- `train_semantic_gpt.py`: train the semantic Transformer.
- `train_semantic_ae.py`: train the frame-wise semantic autoencoder baseline.
- `eval_semantic_ae.py`: evaluate an autoencoder checkpoint on multi-label metrics.
- `semantic_gpt_latent.py`: export Transformer latents on predefined trajectories.
- `semantic_gpt_error_map.py`: train a position decoder from cached latents and plot
  spatial error maps.
- `semantic_gpt_vocab_select.py`: compute channel-by-vocabulary activation heatmaps.
- `semantic_gpt_vocab_selectivity.py`: rank channels by vocabulary selectivity.
- `semantic_gpt_placefield.py`: project high-activation latent channels back onto
  the environment map.
- `make_semantic_degenerate.py`: build controlled semantic aliasing datasets by
  merging object-label columns.

## Installation

Create a Python environment and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

GPU acceleration is optional but recommended for training.

## Notes for Collaboration

Before asking another agent or online model to reason about this project, point it
to this README and the files listed above. Avoid using the legacy `a00`-`a04` and
`a94`-`a99` scripts as source of truth unless explicitly revisiting older
multimodal experiments.
