# Semantic Predictive Coding Pipeline

This repository contains the active semantic-only pipeline for the Twin Mansion
predictive-coding experiments. Data, checkpoints, generated analyses, and legacy
multimodal scripts are intentionally excluded from version control.

An independent RGB-only predictive baseline is now available alongside this
semantic pipeline. See [VISUAL_TRAINING.md](VISUAL_TRAINING.md) for its architecture,
Colab timing/training commands, prediction evaluation and latent exports. Legacy
models are preserved; the new visual entry point is `train_visual_predictive.py`.

## Training Progress

The active visual, SemanticGPT and semantic AE trainers show text progress bars
by default, including when launched with `!python` in Colab. Train/validation bars
show batch counts, running average loss, speed, elapsed time and phase ETA.
Epoch summaries also report wall-clock duration and `ETA(max_epochs)`, estimated
from completed epochs including validation and saving. This estimate excludes
initial setup and may overestimate the run when early stopping triggers.
Visual benchmark warmup and measured steps have separate progress bars.

Add `--no_progress` to hide batch bars while retaining epoch summaries. Display
refreshes are throttled to about once per second; the first batches need to run
before a useful speed/ETA is available. No notebook widget dependency is needed.

New training entry points should reuse `src/training_progress.py`, enable progress
by default, support `--no_progress`, and label each training/validation phase and
epoch. Report phase ETA separately from the estimate to the configured epoch cap.

## Scientific Contract

`SemanticGPT` is a future-prediction model. For a window starting at frame `s`:

```text
input  = semantics[s : s + L]
target = semantics[s + h : s + h + L]
```

The default horizon is `h=1`. At output position `t`, frame-causal attention
allows the latent to use semantic tokens from frames `<=t` only, while its target
is frame `t+h`. Tokens inside one frame are an unordered set and are mutually
visible. Future frames and padding tokens cannot be attended to.

The semantic autoencoder is intentionally different: it reconstructs the current
frame from the current frame and has no temporal input. It is the same-frame
baseline for dilation and latent comparisons.

Neither model uses position, state, actions, room ID, map labels, images, or
visual/semantic factors as input or training targets. State and actions may be
exported beside latents for post-hoc probes only.

## Trail Dataset

The primary dataset is a Unity trail directory:

```text
trail_YYYYMMDD_HHMMSS_FRAMECOUNT/
  semantics.jsonl
  state.npy
  actions.npy
  unity_frames.npy
  episodes.npy
  frame_index.jsonl
  occupancy.npy
  navigation.json
  object_map.csv
  meta.json
  pathN/
    ...
```

`semantics.jsonl` is the authoritative model data source. Each line is one global
frame and should contain atomic strings under a supported field such as
`modelTokens`, `model_tokens`, `tokens`, or `visibleTokens`. A token such as
`S1_core`, `<ANCHOR_0>`, or `prop_bed_08` maps to one ID and is never split on
underscores.

Vocabulary discovery follows this order:

1. `--vocab_path`
2. an explicit semantic vocabulary in `meta.json`
3. a `modelToken`, `token`, or `semanticLabel` column in `object_map.csv`
4. a deterministic scan of `semantics.jsonl`

A scene table with only columns such as `name`, `cx`, `cy`, and `cz` is not used
as the semantic vocabulary. The finalized mapping is saved as
`semantic_vocab.json` beside each checkpoint.

Inspect raw tokens, atomic IDs, decoded IDs, and multi-hot targets before training:

```bash
python scripts/inspect_semantic_tokens.py \
  --data_root data/trail_latest \
  --start_frame 0 \
  --num_frames 5
```

Sliding windows use `stride=1` by default and never cross episode boundaries.
When no separate validation trail is provided, the split is episode-based. A
single-episode dataset falls back to sequence-level splitting with a warning.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train future-predictive SemanticGPT:

```bash
python train_semantic_gpt.py \
  --data_root data/trail_latest \
  --out_dir experiments/semantic_gpt \
  --sequence_length 25 \
  --horizon 1 \
  --stride 1 \
  --device cuda
```

Train the same-frame autoencoder baseline:

```bash
python train_semantic_ae.py \
  --data_root data/trail_latest \
  --out_dir experiments/semantic_ae \
  --sequence_length 25 \
  --stride 1 \
  --device cuda
```

Export SemanticGPT latents and aligned post-hoc metadata:

```bash
python semantic_gpt_latent.py \
  --data_root data/trail_latest \
  --ckpt experiments/semantic_gpt/best.ckpt \
  --out_npz analysis_out/semantic_gpt_latents.npz \
  --device cuda
```

The NPZ contains `z`, `semantics_input`, `semantics_target`, predictions, input
and target frame indices, and episode IDs. When present in the trail, it also
contains state/actions/Unity frame IDs and image paths. Compatibility aliases
`semantics` and `positions` are retained; `positions` is the input state ordered
as `(x, z, yaw)`.

## Smoke Checks

These checks are small and CPU-safe:

```bash
python scripts/smoke_test_atomic_tokenizer.py
python scripts/inspect_semantic_tokens.py --data_root data/trail_latest --num_frames 5
python scripts/smoke_test_semantic_dataset.py
python scripts/smoke_test_frame_causal_mask.py
python scripts/dry_run_train_semantic_gpt.py
python scripts/dry_run_train_semantic_ae.py
```

See `COLAB_TRAINING.md` for a Drive-based Colab workflow.

## Analysis Scripts

- `semantic_gpt_error_map.py`: position-decoder error maps from cached latents.
- `semantic_gpt_vocab_select.py`: channel-by-token activation heatmaps.
- `semantic_gpt_vocab_selectivity.py`: rank token-selective latent channels.
- `semantic_gpt_placefield.py`: project latent activations onto state coordinates.
- `make_semantic_degenerate.py`: create semantic aliasing controls for legacy data.

The old `a00`-`a04` and `a94`-`a99` scripts are not the source of truth for this
pipeline.
