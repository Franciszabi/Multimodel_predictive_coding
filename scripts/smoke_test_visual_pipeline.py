"""CPU regression tests for the independent visual pipeline; no Unity data needed."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.trail_visual_dataset import TrailVisualSequenceDataset, image_relative_path
from src.models.visual_predictive import VisualPredictiveModel
from train_visual_predictive import main as train_main, load_visual_checkpoint
from visual_predictive_latent import main as export_main
from visual_latent_analysis import main as analysis_main
from eval_visual_predictive import main as eval_main


def make_trail(root, frames=32):
    root.mkdir(parents=True)
    records = []
    for i in range(frames):
        folder = f"path{i // 5 + 1}"
        (root / folder).mkdir(exist_ok=True)
        image = np.full((16, 16, 3), (i * 7) % 256, dtype=np.uint8)
        image[:, :i % 16, 1] = 255
        Image.fromarray(image).save(root / folder / f"{i % 5}.png")
        records.append(dict(globalStep=i, episode=i // 16, path=folder, frame=i % 5))
    (root / "frame_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    np.save(root / "episodes.npy", np.arange(frames) // 16)
    np.save(root / "state.npy", np.column_stack((np.arange(frames) % 8, np.arange(frames) // 8, np.zeros(frames))))


class VisualPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.trail = cls.root / "train"
        cls.val = cls.root / "validation"
        make_trail(cls.trail)
        make_trail(cls.val)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_paths_and_shifted_targets(self):
        self.assertEqual(image_relative_path({"path": "path2", "frame": 3}), "path2/3.png")
        self.assertEqual(image_relative_path({"image_path": "path2\\3.png"}), "path2/3.png")
        for value in ("../3.png", "C:\\data\\3.png", "/data/3.png", "path1"):
            with self.assertRaises(ValueError):
                image_relative_path({"image_path": value})
        dataset = TrailVisualSequenceDataset(self.trail, 3, 2, stride=1, image_size=16, return_metadata=True)
        sample = dataset[4]
        self.assertEqual(sample["input_indices"].tolist(), [4, 5, 6])
        self.assertEqual(sample["target_indices"].tolist(), [6, 7, 8])
        torch.testing.assert_close(sample["images"][2], sample["targets"][0])
        self.assertAlmostEqual(float(sample["targets"][2, 0, 0, 0]), 56 / 255, places=6)
        self.assertNotIn(12, dataset.sequence_map)
        self.assertIn(16, dataset.sequence_map)
        ae = TrailVisualSequenceDataset(self.trail, 3, 0, image_size=16)
        torch.testing.assert_close(ae[0]["images"], ae[0]["targets"])

    def test_temporal_split_and_metadata_only(self):
        train = TrailVisualSequenceDataset(self.trail, 3, 2, image_size=16, stride=1, split="train", val_fraction=0.25)
        val = TrailVisualSequenceDataset(self.trail, 3, 2, image_size=16, stride=1, split="val", val_fraction=0.25)
        self.assertLess(max(train.sequence_map) + 3 + 2 - 1, min(val.sequence_map))
        export = TrailVisualSequenceDataset(self.trail, 3, 2, image_size=16, return_targets=False, return_metadata=True)
        self.assertNotIn("targets", export[0])
        self.assertNotIn("state", export[0])

    def test_causality_and_history_in_train_and_eval(self):
        torch.manual_seed(19)
        images = torch.rand(2, 4, 3, 16, 16)
        for positional in ("none", "sinusoidal"):
            model = VisualPredictiveModel(position_encoding=positional)
            for training in (True, False):
                model.train(training)
                with torch.no_grad():
                    predicted = model(images)
                    self.assertEqual(predicted.shape, images.shape)
                    changed = images.clone()
                    changed[:, 2:] = torch.rand_like(changed[:, 2:]) * 3
                    torch.testing.assert_close(predicted[:, :2], model(changed)[:, :2], atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(predicted[:, :2], model(images[:, :2]), atol=2e-5, rtol=2e-5)
                    changed = images.clone()
                    changed[:, 0] = torch.rand_like(changed[:, 0])
                    difference = (model(changed)[:, -1] - predicted[:, -1]).abs().max()
                    self.assertGreater(float(difference), 1e-5)
                    changed = images.clone()
                    changed[1] *= 4
                    torch.testing.assert_close(predicted[0], model(changed)[0], atol=2e-5, rtol=2e-5)
        model.train()
        source = images.clone().requires_grad_(True)
        prediction, layers = model(source, return_latents=True)
        self.assertEqual(set(layers), {"encoder", "temporal_1", "temporal_2", "final"})
        prediction[:, 1].square().mean().backward()
        self.assertEqual(float(source.grad[:, 2:].abs().max()), 0)
        self.assertGreater(float(source.grad[:, :2].abs().max()), 0)
        self.assertTrue(layers["final"].requires_grad)

    def test_ae_independence_and_resolution(self):
        model = VisualPredictiveModel(model_type="autoencoder")
        for training in (True, False):
            model.train(training)
            x = torch.rand(1, 3, 3, 32, 32)
            with torch.no_grad():
                original = model(x)
                x[:, :2] *= 3
                torch.testing.assert_close(original[:, -1], model(x)[:, -1])
                self.assertEqual(original.shape, x.shape)
        with self.assertRaises(ValueError):
            VisualPredictiveModel(num_heads=3)

    def test_train_resume_benchmark_export_analysis(self):
        out = self.root / "run"
        shared = ["--data_root", str(self.trail), "--val_data_root", str(self.val),
                  "--sequence_length", "3", "--image_size", "16", "--stride", "1",
                  "--batch_size", "2", "--limit_train_samples", "4", "--limit_val_samples", "2",
                  "--num_workers", "0", "--device", "cpu", "--amp", "off"]
        train_main(shared + ["--out_dir", str(out), "--epochs", "1"])
        checkpoint = out / "last.ckpt"
        model, payload = load_visual_checkpoint(checkpoint, torch.device("cpu"))
        self.assertEqual(payload["epoch"], 1)
        self.assertFalse(payload["config"]["state_used_as_input"])
        del model, payload
        with self.assertRaises(FileExistsError):
            train_main(shared + ["--out_dir", str(out), "--epochs", "1"])
        train_main(shared + ["--out_dir", str(out), "--epochs", "2", "--resume", str(checkpoint)])
        logs = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
        self.assertEqual([entry["epoch"] for entry in logs], [1, 2])
        self.assertGreater(logs[0]["validation"]["persistence_mse"], 0)
        self.assertTrue((out / "best_prediction.png").is_file())
        evaluated = eval_main(["--ckpt", str(checkpoint), "--data_root", str(self.val),
                               "--out_dir", str(self.root / "evaluation"), "--num_workers", "0", "--limit_samples", "2"])
        self.assertAlmostEqual(evaluated["mse"], logs[-1]["validation"]["mse"], places=5)
        report = train_main(shared + ["--out_dir", str(self.root / "benchmark"), "--benchmark_steps", "2", "--warmup_steps", "1"])
        self.assertEqual(report["samples"], 4)
        self.assertGreater(report["samples_per_second"], 0)
        ae_out = self.root / "ae"
        train_main(shared + ["--model_type", "autoencoder", "--out_dir", str(ae_out), "--epochs", "1"])
        _, ae = load_visual_checkpoint(ae_out / "last.ckpt", torch.device("cpu"))
        self.assertEqual(ae["config"]["horizon"], 0)
        for name, root in (("train", self.trail), ("test", self.val)):
            export_main(["--ckpt", str(checkpoint), "--data_root", str(root), "--out_npz", str(self.root / f"{name}.npz"),
                         "--num_workers", "0", "--device", "cpu", "--limit_samples", "4", "--batch_size", "2"])
        with np.load(self.root / "test.npz", allow_pickle=False) as result:
            self.assertEqual(result["latents"].shape, (4, 128, 2, 2))
            self.assertEqual(result["input_indices"][0], 2)
            self.assertEqual(result["target_indices"][0], 3)
            np.testing.assert_equal(result["state"][0], [2, 0, 0])
        analysis = analysis_main(["--npz", str(self.root / "test.npz"), "--train_npz", str(self.root / "train.npz"),
                                  "--out_dir", str(self.root / "analysis"), "--bins", "3", "--min_occupancy", "1", "--units", "2"])
        self.assertIn("position_probe", analysis)
        with self.assertRaises(ValueError):
            analysis_main(["--npz", str(self.root / "test.npz"), "--train_npz", str(self.root / "test.npz"),
                           "--out_dir", str(self.root / "overlap"), "--units", "1"])

    def test_windows_worker(self):
        dataset = TrailVisualSequenceDataset(self.trail, 3, 1, image_size=16)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=1)
        batches = list(loader)
        self.assertGreater(len(batches), 0)
        self.assertEqual(tuple(batches[0]["images"].shape), (2, 3, 3, 16, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
