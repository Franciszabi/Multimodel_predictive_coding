"""Atomic semantic token vocabulary and deterministic vocabulary discovery."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DEFAULT_EMPTY_TOKEN = "<NO_SEMANTIC_VISIBLE>"


class AtomicSemanticTokenizer:
    """Map each complete semantic string to exactly one token id."""

    pad_id = 0
    unk_id = 1
    tokenizer_type = "atomic"

    def __init__(self, tokens: Sequence[str]) -> None:
        clean_tokens: list[str] = []
        seen: set[str] = set()
        for raw_token in tokens:
            token = str(raw_token).strip()
            if not token or token in {PAD_TOKEN, UNK_TOKEN} or token in seen:
                continue
            seen.add(token)
            clean_tokens.append(token)
        if not clean_tokens:
            raise ValueError("Atomic semantic vocabulary is empty.")

        self.tokens = clean_tokens
        self.token_to_class_index = {token: i for i, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens) + 2

    @property
    def num_classes(self) -> int:
        return len(self.tokens)

    def encode_token(self, token: str) -> int:
        class_index = self.token_to_class_index.get(str(token).strip())
        return self.unk_id if class_index is None else class_index + 2

    def encode_frame(
        self,
        tokens: Sequence[str],
        max_tokens: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        token_ids = np.full(max_tokens, self.pad_id, dtype=np.int64)
        token_mask = np.zeros(max_tokens, dtype=np.bool_)
        encoded = [self.encode_token(token) for token in tokens[:max_tokens]]
        if encoded:
            token_ids[: len(encoded)] = np.asarray(encoded, dtype=np.int64)
            token_mask[: len(encoded)] = True
        return token_ids, token_mask

    def frame_to_multihot(self, tokens: Sequence[str]) -> np.ndarray:
        target = np.zeros(self.num_classes, dtype=np.float32)
        for token in tokens:
            class_index = self.token_to_class_index.get(str(token).strip())
            if class_index is not None:
                target[class_index] = 1.0
        return target

    def decode_token_id(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id == self.pad_id:
            return PAD_TOKEN
        if token_id == self.unk_id:
            return UNK_TOKEN
        class_index = token_id - 2
        if 0 <= class_index < len(self.tokens):
            return self.tokens[class_index]
        return UNK_TOKEN

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "token_to_class_index": self.token_to_class_index,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "tokenizer_type": self.tokenizer_type,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AtomicSemanticTokenizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("tokenizer_type", "atomic") != "atomic":
            raise ValueError(f"Vocabulary at {path} is not atomic.")
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError(f"Vocabulary at {path} must contain a 'tokens' list.")
        tokenizer = cls(tokens)
        expected = payload.get("token_to_class_index")
        if expected is not None and expected != tokenizer.token_to_class_index:
            raise ValueError(f"Vocabulary mapping in {path} is inconsistent with token order.")
        return tokenizer


def _unique_strings(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _tokens_from_meta(meta_path: Path) -> list[str]:
    if not meta_path.exists():
        return []
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    candidate_keys = (
        "semantic_vocabulary",
        "semanticVocabulary",
        "semantic_vocab",
        "semantic_tokens",
        "model_tokens",
        "modelTokens",
        "token_list",
        "vocabulary",
        "tokens",
    )
    containers = [meta]
    for key in ("semantics", "semantic", "collection"):
        value = meta.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key in candidate_keys:
            value = container.get(key)
            if isinstance(value, list):
                return _unique_strings(value)
    return []


def _tokens_from_object_map(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []
    candidate_columns = (
        "modelToken",
        "model_token",
        "token",
        "semanticLabel",
        "semantic_label",
    )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        selected = next((name for name in candidate_columns if name in fieldnames), None)
        if selected is None:
            return []
        return _unique_strings(row.get(selected, "") for row in reader)


def _tokens_from_semantics(semantics_path: Path) -> list[str]:
    from .trail_semantic_dataset import load_semantic_jsonl

    frames = load_semantic_jsonl(semantics_path, include_empty_token=False)
    return sorted({token for frame in frames for token in frame})


def build_atomic_tokenizer(
    data_root: str | Path,
    vocab_path: str | Path | None = None,
    include_empty_token: bool = True,
    empty_token: str = DEFAULT_EMPTY_TOKEN,
) -> tuple[AtomicSemanticTokenizer, str]:
    """Build vocabulary using the documented priority order."""

    data_root = Path(data_root)
    if vocab_path:
        tokenizer = AtomicSemanticTokenizer.load(vocab_path)
        source = str(Path(vocab_path))
    else:
        tokens = _tokens_from_meta(data_root / "meta.json")
        source = "meta.json"
        if not tokens:
            tokens = _tokens_from_object_map(data_root / "object_map.csv")
            source = "object_map.csv"
        if not tokens:
            tokens = _tokens_from_semantics(data_root / "semantics.jsonl")
            source = "semantics.jsonl"
        if source != "meta.json":
            tokens = sorted(set(tokens))
        tokenizer = AtomicSemanticTokenizer(tokens)

    if include_empty_token and empty_token not in tokenizer.token_to_class_index:
        tokenizer = AtomicSemanticTokenizer([*tokenizer.tokens, empty_token])
    return tokenizer, source
