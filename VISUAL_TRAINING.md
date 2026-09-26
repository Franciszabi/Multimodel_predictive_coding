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

Warmup and benchmark progress bars are enabled by default. They report batch
counts, average loss, speed and ETA, including in Colab `!python` output. Use
`--no_progress` to disable display for timing comparisons or redirected logs.

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

Training and validation each show `epoch/max_epochs`, batch progress and phase
ETA. At epoch end, `ETA(max_epochs)` estimates remaining wall-clock time including
validation and checkpoint saving; early stopping may finish sooner. After resume,
the estimate uses only epochs completed in the current run. Timing fields are also
saved in `train_log.jsonl`. `--no_progress` keeps these epoch summaries and logs.

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

## Latents and Spatial Analysis

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
  --out_dir "{RUN}/place_fields_q90_val" \
  --quantile 0.9 --bins 30 --min_occupancy 1 --units 16
```

Full feature maps are saved by default: `[N,128,8,8]` at 64px. `--pool spatial_mean`
reduces storage to `[N,128,1,1]`. Both produce the same channel-mean representation
for the place-field analysis, but pooling discards the feature-map layout needed
by the James position decoder. Use `--pool none` (default) for position decoding.
Export stops before allocating more than `--max_output_mb` (default 2048 MiB)
for the latent array. Total process memory also includes model and image batches.

### Reference-Style Place Fields

`visual_latent_analysis.py` follows the calculation in the original
[PlaceFields implementation](https://github.com/jgornet/predictive-coding-recovers-maps/blob/main/predictive_coding/analysis.py),
also used by the legacy `src/analysis.py`:

1. Average each channel over its feature-map axes: `[N,C,H,W] -> [N,C]`.
2. Calculate each channel's 90th percentile over all samples in `--npz`.
3. Select samples with activation **strictly greater than** that threshold.
   Ties can yield fewer than 10% active samples; a constant channel has none.
4. Histogram the selected samples' x/z positions. These are raw high-activation
   counts, NOT average activations and NOT divided by occupancy.
5. Define the binary field as `count > 0`. Fit a 2D Gaussian to the selected
   continuous coordinates using their mean and sample covariance. Save the
   public code's `approx_areas = pi * sqrt(det(covariance))` (one-sigma ellipse).

The one-sigma area is not the paper text's fixed-density `P >= 0.0005` region.
Gaussian densities are evaluated at the configured bin centers for display;
the reference uses a finer display mesh. No covariance regularization is added:
fewer than three active samples or a singular covariance gives an unavailable
Gaussian, recorded in the report, while the raw and binary fields remain valid.

All channels are computed and saved. `--units 16` selects fixed evenly spaced
**channels only for plotting**, not flattened feature-map elements or top-ranked
fields. Use `--units 128` to plot all channels of the current visual model.
`--bins 30` uses 30 bins per axis; `--bins NX NZ` accepts different axis sizes.
For comparable maps, specify the same `--spatial_range XMIN XMAX ZMIN ZMAX` and
bins across exports. Otherwise exported map bounds are used when available,
falling back to this export's position range.
World coordinates are not hardcoded to the original Minecraft environment.

`--min_occupancy` now defaults to 1, suitable for a scan with one exported sample
per location. It is a **display-only mask**; it never changes thresholds, raw
counts, binary fields, Gaussian fitting or coverage statistics. Empty bins are
not labeled as obstacles, since this export alone cannot distinguish obstacles
from locations that were not sampled.

Outputs in a new analysis directory:

- `place_fields.npz`: all channel means, thresholds, active-sample masks, high
  activation counts, binary fields, occupancy, bin edges, Gaussian parameters,
  fit status, densities, one-sigma areas and coverage counts.
- `place_fields.png`: high-activation count maps with occupancy.
- `place_fields_binary.png`: binary support of each channel's field.
- `place_fields_gaussian.png`: fitted Gaussian density maps at sampled bins.
- `place_field_statistics.png`: ellipse areas, fields per covered bin, and bins
  per channel (including empty channels).
- `report.json`: method, quantile, bounds, sampling metadata and diagnostics.

Previous `activation_maps.*` outputs are not overwritten or silently relabeled.
Existing `latent_train.npz`/`latent_val.npz` exports can be reused without training
or exporting again; only rerun analysis into a new output directory.

### Room and Obstacle Overlays

Both place-field plots and decoder error maps read `map.json`, `navigation.json`
and `occupancy.npy`. They auto-detect these files under the latent export's
`data_root`; use `--map_root "{SCAN_ROOT}"` if files moved or use another compatible
export of the **same map**. Map coordinates are x horizontally, z vertically.
Gray marks exported obstacle occupancy (which can include agent-radius inflation).
Dashed room cells use the exported room centers and spacing, with room IDs and
door markers. These are room-layout guides, not exact wall/door polygons. No
5-by-5 layout or fixed map size is hardcoded. The overlay never enters the model.

### Separate James Position Decoder

The previous Ridge probe and its `--train_npz`/`--ridge_alpha` options were removed
from `visual_latent_analysis.py`. Position decoding now uses a separate script:

```python
!python visual_position_decoder.py --train_npz "{RUN}/latent_train.npz" \
  --npz "{RUN}/latent_val.npz" --out_dir "{RUN}/position_decoder_val" \
  --map_root "{SCAN_ROOT}" --device cuda
```

The visual model is frozen. Only the auxiliary decoder learns coordinates from
saved **full `[N,128,8,8]`** latents: Conv2d(128,256,3,padding=1), MaxPool2d(2),
Flatten, Linear(4096,64), ReLU, Linear(64,2). No channel averaging, scaler or Ridge.
The default follows the public `PositionDecoder` source: float32, batch 512,
AdamW lr=1e-4 / weight_decay=0.01, 8000 epochs, StepLR(4000,0.1), coordinates /30.
Predictions are multiplied by 30 before Euclidean errors are measured. As in the
source, shuffled remainders are dropped; for fewer than 512 samples we use one
partial batch so training still occurs. The source's mis-scaled diagnostic log
is corrected. A progress bar shows epoch progress/ETA. Weights/logs save every
100 epochs and at the end (`--save_every` changes checkpoint frequency).

The paper prose instead reports 1000+1000 epochs and a ReLU after convolution.
This implementation matches the **public code**, not both conflicting versions.
For a short execution check set `--epochs 200 --lr_step 100`, but label this as a
shortened budget, not the source's training schedule. There is no validation-
driven checkpoint selection or early stopping: the final decoder is evaluated.

Outputs: `decoder.ckpt` (weights, optimizer, schedule, configuration and training
provenance), `decoder_train_log.jsonl`, `position_predictions.npz` (predicted/true
coordinates and per-sample error), `error_map.png`, `error_map.npz` (hexagon means),
and `report.json`. The error map uses mean Euclidean error per hexagon, as in
the source; x/z axes follow Unity rather than the source's rotated Minecraft map.
Use a common `--error_vmax` across model comparisons; by default colors auto-scale.

Evaluate saved weights without retraining:

```python
!python visual_position_decoder.py --ckpt "{RUN}/position_decoder_val/decoder.ckpt" \
  --npz "{RUN}/latent_val.npz" --out_dir "{RUN}/position_decoder_val_rerender" \
  --map_root "{SCAN_ROOT}" --device cuda
```

`--train_npz` fits only that export and evaluates `--npz`. Overlapping same-trail
frame supports are rejected. Existing train25/val25 exports can be reused.
The public `notebooks/predictive_coding.ipynb` instead loads
`predictive-coder-environment-images.npy`, fits the decoder on those latents,
then plots error on the **same** latents. The notebook calls these data a visual
validation dataset, but they are not held out from the auxiliary decoder. Reproduce that protocol
deliberately with `--fit_on_eval`; outputs are labeled `same_sample_fit`.

### Grid Scan Inference

Training L=25 is a context length, not a fixed architectural input dimension.
For James-style independent local contexts use each complete anchor's ten shifts:

```python
!python visual_predictive_latent.py --ckpt "{RUN}/best.ckpt" --data_root "{SCAN_ROOT}" \
  --out_npz "{RUN}/latent_scan_groups.npz" --scan_groups --layer final --device cuda
!python visual_latent_analysis.py --npz "{RUN}/latent_scan_groups.npz" \
  --out_dir "{RUN}/place_fields_scan" --map_root "{SCAN_ROOT}" --units 16 --bins 64
!python visual_position_decoder.py --npz "{RUN}/latent_scan_groups.npz" --fit_on_eval \
  --out_dir "{RUN}/position_decoder_scan_fit" --map_root "{SCAN_ROOT}" --device cuda
```

`--scan_groups` reads complete groups, sorts shift IDs and validates image/index
alignment. It exports one last-input latent per anchor and labels it with the
**actual last shift's position**, not the anchor's first position. It needs no
future target. It records actual L=10 separately from trained L=25; it neither
repeats frames nor pads to 25. `pathN` folders remain storage chunks only.

Concatenating A's ten frames, B's ten and C's first five is also computationally
valid. Omitting `--scan_groups` retains the existing checkpoint-length windows
and `--stride` behavior, but marks them `scan_collection_windows`. This includes
artificial scan-order transitions and row jumps; its fields are conditional on
that history, not the same experiment as independent local contexts. L=25
permits longer history, but does not guarantee it is informative or in-distribution.
Compare protocols separately. By default the decoder rejects unmatched context
lengths or sampling protocols across fit/evaluation exports. To intentionally
test a trajectory-trained decoder on scan groups, reuse its saved weights:

```python
!python visual_position_decoder.py --ckpt "{RUN}/position_decoder_val/decoder.ckpt" \
  --npz "{RUN}/latent_scan_groups.npz" --out_dir "{RUN}/position_decoder_scan_transfer" \
  --allow_context_shift --map_root "{SCAN_ROOT}" --device cuda
```

This performs inference only, without fitting on scan coordinates. The same flag
also works with `--train_npz` when a new decoder needs to be fitted. Outputs are
labeled `separate_export_context_shift` for independent exports and record the
actual train/evaluation context differences in `report.json`. For train L=25
versus scan L=10, this tests transfer across both trajectories and input context;
it is not a matched-context generalization comparison. Visual checkpoint, layer,
pooling, horizon, image size and latent shape checks remain strict, as does the
overlap guard when fitting a decoder. Do not edit NPZ metadata to bypass checks
or concatenate teleports just to force matching lengths. Coordinates remain
labels, never visual-model inputs.

These checks do not establish significant place/grid cells or path integration.
For those claims, add trajectory-aware nulls, cross-trajectory reliability,
matched AE/current-frame controls, multiple seeds and direction/history controls.
Overlapping frame supports from the same trail are rejected for probe fitting;
duplicate trajectories copied to different folders cannot be detected automatically.

## Local Verification

```bash
python scripts/smoke_test_visual_pipeline.py
```

Tests cover training/eval causality, history sensitivity, batch independence,
image/target indexing, cross-folder windows, episode boundaries, temporal splits,
AE independence, checkpoint resume, timing, evaluation, latent alignment,
scan grouping, map overlays and the CNN position decoder.
On Windows, if Anaconda reports duplicate OpenMP runtimes, run the test in a clean
environment or set `MKL_THREADING_LAYER=SEQUENTIAL` for that process; do not enable
the unsafe `KMP_DUPLICATE_LIB_OK` workaround.

Initial local checks passed on PyTorch 2.6.0 CPU: six regression tests and a
real-data smoke train/validation/export run. A five-step CPU benchmark after two
warmup steps measured about 1.09 seconds/step at 64px, L=25, batch=2, workers=0,
and two Torch CPU threads. This is not a GPU speed estimate or a convergence run;
no paper-level result has been reproduced by these checks.
