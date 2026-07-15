"""Verify that semantic strings remain indivisible atomic tokens."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.semantic_tokenizer import AtomicSemanticTokenizer, build_atomic_tokenizer


def main() -> None:
    tokens = ["S1_core", "<SEM_B>", "prop_bed_08", "<DOORWAY>"]
    tokenizer = AtomicSemanticTokenizer(tokens)
    ids, mask = tokenizer.encode_frame(tokens, max_tokens=8)
    assert int(mask.sum()) == len(tokens)
    assert len(set(ids[mask].tolist())) == len(tokens)
    assert [tokenizer.decode_token_id(value) for value in ids[mask]] == tokens
    assert tokenizer.vocab_size == len(tokens) + 2

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        vocab_path = root / "semantic_vocab.json"
        tokenizer.save(vocab_path)
        reloaded = AtomicSemanticTokenizer.load(vocab_path)
        assert reloaded.tokens == tokens
        (root / "meta.json").write_text("{}\n", encoding="utf-8")
        with (root / "object_map.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "cx", "cy", "cz"])
            writer.writeheader()
            writer.writerow({"name": "SemanticCue_R01_S1_left", "cx": 1, "cy": 0, "cz": 2})
        with (root / "semantics.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"modelTokens": ["S1_core", "<DOORWAY>"]}) + "\n")
        discovered, source = build_atomic_tokenizer(root)
        assert source == "semantics.jsonl"
        assert "S1_core" in discovered.token_to_class_index
        assert "SemanticCue_R01_S1_left" not in discovered.token_to_class_index
    print("PASS: all semantic strings map to one atomic token id")


if __name__ == "__main__":
    main()
