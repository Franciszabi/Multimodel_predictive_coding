# Independent Visual Baseline V1

The legacy models and semantic pipeline are unchanged. This new pipeline has two
initial purposes: measure real training cost and test whether future prediction
and useful spatial representations emerge in the current Unity environment.
It does not implement multimodal fusion yet.

## Architecture and Scientific Contract

```text
RGB [B,L,3,64,64], range [0,1]
  -> per-frame residual CNN encoder [B,L,128,8,8]
  -> two frame-causal attention + pointwise FFN blocks
  -> latent [B,L,128,8,8]
  -> per-frame residual CNN decoder [B,L,3,64,64]
```

- Default: 8 heads, 2 temporal layers, approximately 1.52 million parameters.
- Attention mixes frames, not individual spatial patches. Each head's feature
  dimension is `(128 / heads) * Hlatent * Wlatent`; at 64px and 8 heads it is 1024.
  Do not assume FlashAttention will be available for this head dimension.
- No pretrained weights, actions, coordinates, semantic labels or room IDs.
- The CNN and decoder topology follow the `[2,2]` residual-block configuration
  in [Gornet and Thomson's code](https://github.com/jgornet/predictive-coding-recovers-maps).
  The associated [paper](https://www.nature.com/articles/s42256-024-00863-1)
  is the scientific reference, not a promise of identical results here.
- This is a **causal adaptation**, not an exact reproduction: BatchNorm is
  replaced by per-frame GroupNorm; temporal LayerNorm excludes the time axis.
  Both choices prevent future frames from entering normalization statistics.
  PyTorch attention receives `[B,heads,L,D]` and `is_causal=True`.
- Positional encoding is off by default, matching the inspected legacy forward
  path. `--position_encoding sinusoidal` is a separately labeled ablation.
- Default optimizer is AdamW, not the original training recipe. SGD and OneCycle
  are optional; preprocessing, normalization, trajectory statistics and training
  budget must be matched before making a strict replication claim.
- No new temporal convolution or fusion adapters are introduced in this baseline.

For a window starting at `s`, inputs are `I[s:s+L]` and targets are
`I[s+h:s+h+L]`. Output position `j` sees only inputs `0..j` and predicts `s+j+h`.
The final output **does contribute to loss**: its target is loaded separately
from the index, even though that future image is not a model input at that time.
Increasing `h` shifts targets; it does not require decoding all intervening frames.

Loss is MSE over all L predictions. Validation also reports last-position MSE,
where every sample has the same L-frame context, and the persistence baseline
`prediction(t+h) = image(t)`. Positive `skill_vs_persistence = 1-MSE/MSE_copy`
means improvement over copying. Images are not clamped for loss or metrics;
only preview displays are clamped to [0,1]. Scores are not directly comparable
to legacy losses computed with different normalization.
All-position metrics count window occurrences, which may overlap; last-position
metrics use one prediction per window and are the cleaner fixed-context comparison.

## Data

Point `--data_root` at a trail containing `frame_index.jsonl`, `episodes.npy`
and `pathN/*.png`. The index determines order, not filesystem sorting.
`{"globalStep":0,"episode":0,"path":"path1","frame":0}` resolves to `path1/0.png`.
`pathN` directories are storage chunks: windows may cross them, but not episode
boundaries. Missing images and inconsistent episode metadata raise errors.

Supply an independent `--val_data_root` when possible. Otherwise, raw frames
are split chronologically before windows are built, including target support.
This avoids train/validation overlap from neighboring windows. Keep a separate
test trail for final claims after using validation for model selection.
Training does not read `state.npy`; latent export may attach its `[x,z,yaw]`
columns for post-hoc analysis only. `semantics.jsonl` is not needed.

## Colab: Setup and Timing

Run after the new files have been committed/pushed and pulled into Colab.
Keep extracted images on `/content`, not mounted Drive, for timing. These paths
match the previously discussed train/validation archives. A new runtime needs
the data copied/extracted to `/content` again.

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Multimodel_predictive_coding
!git pull --ff-only origin main
%pip install -r requirements.txt

from pathlib import Path
TRAIN_ROOT = '/content/pred_data/train_200000/trail_20260715_052524_200000'
VAL_ROOT = '/content/pred_data/validation_100000/trail_20260715_044615_100000'
RUN = '/content/drive/MyDrive/pred_train_data/experiments/visual_pc_v1_h1'
for root in (TRAIN_ROOT, VAL_ROOT):
    assert (Path(root) / 'frame_index.jsonl').is_file(), root
```

First measure 50 optimizer steps after 5 warmup steps. This does not save trained
weights. Use a different output directory for each benchmark configuration.

```python
!python train_visual_predictive.py --data_root "{TRAIN_ROOT}" --val_data_root "{VAL_ROOT}" \
  --out_dir "{RUN}_bench_b8_w2" --sequence_length 25 --horizon 1 --stride 5 \
  --image_size 64 --batch_size 8 --num_workers 2 --device cuda --amp auto \
  --warmup_steps 5 --benchmark_steps 50
```

`benchmark.json` records milliseconds/step, windows/second, input-frame
occurrences/second, data wait, peak allocated GPU memory, GPU name and an
estimated training-only epoch duration. Loader waits, host-to-device transfer,
forward, backward and optimizer are included; validation/checkpoint I/O are not.
Warmup is separate. The estimate is approximate; a full epoch is the confirmation.
Repeated/overlapping windows mean frame occurrences are not unique decoded images.

Compare batch sizes 8/16 and worker counts 0/2/4 using fresh output directories.
If GPU memory runs out, reduce batch size first. Keep L, h and image size fixed
while comparing timing. `--cache_frames N` enables a per-worker float32 LRU cache;
leave it at 0 initially. AMP auto uses bf16 on supported CUDA hardware, otherwise
fp16; CPU uses fp32. Check `skipped_optimizer_steps` when using fp16.

## Training and Evaluation

```python
!python train_visual_predictive.py --data_root "{TRAIN_ROOT}" --val_data_root "{VAL_ROOT}" \
  --out_dir "{RUN}" --sequence_length 25 --horizon 1 --stride 5 \
  --image_size 64 --batch_size 8 --num_workers 2 --epochs 40 \
  --optimizer adamw --lr 0.0003 --weight_decay 0.01 \
  --early_stopping_patience 10 --device cuda --amp auto
```

Outputs: `config.json`, `train_log.jsonl`, `last.ckpt`, `best.ckpt` and
`best_prediction.png` (input/prediction/true future). Epoch logs separate training,
validation and save times. An existing nonempty output directory is refused.
Use a new directory for every changed horizon, model or training recipe.

To resume, repeat the same training command with `--resume "{RUN}/last.ckpt"`.
For the default constant-LR schedule, `--epochs` can be increased; it is the total
desired epoch count, not additional epochs. OneCycle runs require the originally
configured total. Resume restores model, optimizer, scaler and Torch RNG state;
bitwise reproducibility across hardware/worker changes is not guaranteed.

Evaluation without further training:

```python
!python eval_visual_predictive.py --ckpt "{RUN}/best.ckpt" --data_root "{VAL_ROOT}" \
  --out_dir "{RUN}/eval_validation" --batch_size 8 --num_workers 2 --device cuda
```

Suggested controls, each in its own output directory:

- Single-frame predictor: `--model_type predictive --sequence_length 1 --horizon 1`.
  This measures what current appearance alone can predict without temporal history.
- Same-frame AE: `--model_type autoencoder --horizon 0 --sequence_length 25`.
  It uses the same CNN encoder/decoder but no temporal blocks. Frames are independent;
  L is only batching convenience. Copy loss is zero for AE and copy skill is undefined.
- Main temporal predictor: `L=25, h=1`. Vary one architectural choice at a time.

Use identical evaluated frame IDs, image preprocessing and training budgets when
comparing these controls. Their default window counts differ with L and h.

## Latents and Initial Spatial Checks

Export the last input position only, giving a fixed context length and one latent
per sampled window. `final` is after the second temporal FFN/normalization and
before the image decoder. Other choices are `encoder`, `temporal_1`, `temporal_2`.
AE has `encoder` and `final` only. No layer is assumed to be the most informative.

```python
!python visual_predictive_latent.py --ckpt "{RUN}/best.ckpt" --data_root "{TRAIN_ROOT}" \
  --out_npz "{RUN}/latent_train.npz" --layer final --stride 25 --device cuda
!python visual_predictive_latent.py --ckpt "{RUN}/best.ckpt" --data_root "{VAL_ROOT}" \
  --out_npz "{RUN}/latent_val.npz" --layer final --stride 25 --device cuda
!python visual_latent_analysis.py --npz "{RUN}/latent_val.npz" \
  --train_npz "{RUN}/latent_train.npz" --out_dir "{RUN}/spatial_analysis" \
  --bins 30 --min_occupancy 3 --units 16
```

Full feature maps are saved by default: `[N,128,8,8]` at 64px. `--pool spatial_mean`
reduces storage to `[N,128,1,1]`, but changes the representation being analyzed.
Export stops before allocating more than `--max_output_mb` (default 2048 MiB)
for the latent array. Total process memory also includes model and image batches.

Activation maps divide summed activation by occupancy and mask poorly sampled
bins. The plotted units are fixed evenly spaced flattened indices, not cherry-picked.
The optional Ridge probe uses 128 spatial-mean features, with scaler and weights
fitted only on the training export. It predicts x,z and reports distance, R2 and
a train-mean-position baseline on the evaluation export. It is an initial probe,
not a replication of all analysis methods in the paper.

These checks do not establish significant place/grid cells or path integration.
For those claims, add trajectory-aware nulls, cross-trajectory reliability,
matched AE/current-frame controls, multiple seeds and the paper's exact metrics.
Overlapping frame supports from the same trail are rejected for probe fitting;
duplicate trajectories copied to different folders cannot be detected automatically.

## Local Verification

```bash
python scripts/smoke_test_visual_pipeline.py
```

Tests cover training/eval causality, history sensitivity, batch independence,
image/target indexing, cross-folder windows, episode boundaries, temporal splits,
AE independence, checkpoint resume, timing, evaluation, latent alignment and probes.
On Windows, if Anaconda reports duplicate OpenMP runtimes, run the test in a clean
environment or set `MKL_THREADING_LAYER=SEQUENTIAL` for that process; do not enable
the unsafe `KMP_DUPLICATE_LIB_OK` workaround.

Initial local checks passed on PyTorch 2.6.0 CPU: six regression tests and a
real-data smoke train/validation/export run. A five-step CPU benchmark after two
warmup steps measured about 1.09 seconds/step at 64px, L=25, batch=2, workers=0,
and two Torch CPU threads. This is not a GPU speed estimate or a convergence run;
no paper-level result has been reproduced by these checks.
