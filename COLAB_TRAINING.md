# Colab Training

Use a GPU runtime in Colab. Keep the repository under `/content` for faster code
access and keep datasets/checkpoints in Google Drive so they survive a runtime
restart.

## 1. Clone and mount Drive

```python
!git clone https://github.com/Franciszabi/Multimodel_predictive_coding.git
%cd /content/Multimodel_predictive_coding

from google.colab import drive
drive.mount("/content/drive")
```

For an existing clone, use `%cd` followed by `!git pull` instead of cloning it
again.

## 2. Install dependencies

```python
!pip install -q -r requirements.txt
!nvidia-smi
```

## 3. Point to the trail dataset

The path must be the directory that directly contains `semantics.jsonl`, not a
parent directory and not a `pathN` child.

```python
DATA_ROOT = "/content/drive/MyDrive/pred_coding/data/trail_YYYYMMDD_HHMMSS_FRAMECOUNT"
!ls -lh "$DATA_ROOT"
```

If the uploaded data is a zip file:

```python
ZIP_PATH = "/content/drive/MyDrive/pred_coding/data/trail_dataset.zip"
!mkdir -p /content/pred_data
!unzip -q "$ZIP_PATH" -d /content/pred_data
!find /content/pred_data -name semantics.jsonl -print
```

Set `DATA_ROOT` to the folder printed immediately above `semantics.jsonl`.

Inspect a few raw frames and their atomic token IDs before training:

```python
!python scripts/inspect_semantic_tokens.py \
  --data_root "$DATA_ROOT" \
  --start_frame 0 \
  --num_frames 5
```

## 4. Train SemanticGPT

```python
!python train_semantic_gpt.py \
  --data_root "$DATA_ROOT" \
  --out_dir /content/drive/MyDrive/pred_coding/experiments/semantic_gpt_stage1 \
  --sequence_length 25 \
  --horizon 1 \
  --stride 1 \
  --batch_size 32 \
  --epochs 80 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 4 \
  --device cuda
```

This trains only on semantic tokens. State, actions, room/map metadata, and images
are not passed to the model or loss.

## 5. Export aligned latents

```python
!python semantic_gpt_latent.py \
  --data_root "$DATA_ROOT" \
  --ckpt /content/drive/MyDrive/pred_coding/experiments/semantic_gpt_stage1/best.ckpt \
  --out_npz /content/drive/MyDrive/pred_coding/analysis_out/semantic_gpt_stage1_latents.npz \
  --sequence_length 25 \
  --horizon 1 \
  --stride 1 \
  --device cuda
```

The exporter loads `semantic_vocab.json` and model dimensions from the checkpoint
directory. State and actions, when available, are copied only into the output NPZ
for post-hoc analysis.

## 6. Optional AE baseline

```python
!python train_semantic_ae.py \
  --data_root "$DATA_ROOT" \
  --out_dir /content/drive/MyDrive/pred_coding/experiments/semantic_ae_stage1 \
  --sequence_length 25 \
  --stride 1 \
  --batch_size 32 \
  --epochs 80 \
  --device cuda
```

The AE intentionally reconstructs each current frame independently. It is not a
future-prediction model.
