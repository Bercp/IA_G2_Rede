"""Compact memory-mapped cache; legal masks are built only for each batch."""
import argparse
import gzip
import json
import time
from pathlib import Path

import chess
import numpy as np
import torch

from .common import ROOT, save_json, sha256
from .encoding import ACTIONS, VERSION, compact_board, legal_ids, move_to_id


def prepare(source, output, limit=None):
    source, output = Path(source), Path(output)
    fingerprints = {s: sha256(source / f"{s}.jsonl.gz") for s in ("train", "validation", "test")}
    signature = {"encoding": VERSION, "source_sha256": fingerprints, "limit_per_split": limit}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in signature.items()):
            print("Cache já preparado e compatível.", flush=True)
            return
        raise ValueError("Cache incompatível. Escolha outro --output para preservar o cache existente.")
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    counts = {}
    start = time.monotonic()
    for split in fingerprints:
        n = report["moves_by_split"][split]
        n = min(n, limit) if limit else n
        folder = output / split
        folder.mkdir(exist_ok=True)
        shapes = {"boards": ((n, 18, 8, 8), "uint8"), "clocks": ((n,), "uint8"),
                  "targets": ((n,), "int64"), "plies": ((n,), "int32"), "offsets": ((n + 1,), "int64")}
        arrays = {key: np.lib.format.open_memmap(folder / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
                  for key, (shape, dtype) in shapes.items()}
        arrays["offsets"][0] = 0
        legal_parts = []
        processed = 0
        with gzip.open(source / f"{split}.jsonl.gz", "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                row = json.loads(line)
                board = chess.Board(row["fen"])
                if not board.is_valid() or row["split"] != split or row["player_id"] != "levon_aronian":
                    raise ValueError(f"Invalid record {split}:{i}")
                if (board.turn == chess.WHITE) != (row["color"] == "white"):
                    raise ValueError("Player color does not match position")
                target = move_to_id(chess.Move.from_uci(row["move_uci"]))
                legal = legal_ids(board)
                if target not in legal:
                    raise ValueError(f"Illegal label {split}:{i}")
                arrays["boards"][i], arrays["clocks"][i] = compact_board(board)
                arrays["targets"][i] = target
                arrays["plies"][i] = row["ply"]
                arrays["offsets"][i + 1] = arrays["offsets"][i] + len(legal)
                legal_parts.append(legal.astype(np.uint16))
                processed += 1
                if processed % 20000 == 0:
                    print(f"Cache {split}: {processed}/{n} ({time.monotonic()-start:.0f}s)", flush=True)
        if processed != n or not n:
            raise ValueError(f"Incorrect or empty source count for {split}")
        np.save(folder / "legal.npy", np.concatenate(legal_parts))
        for array in arrays.values():
            array.flush()
        del arrays
        counts[split] = n
        print(f"Cache {split}: concluído, {n} posições", flush=True)
    save_json(manifest_path, {**signature, "counts": counts, "diagnostic_only": limit is not None})


class CachedDataset:
    def __init__(self, root, split):
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["encoding"] != VERSION:
            raise ValueError("Unsupported cache encoding")
        self.arrays = {key: np.load(root / split / f"{key}.npy", mmap_mode="r")
                       for key in ("boards", "clocks", "targets", "plies", "offsets", "legal")}
        self.split = split

    def __len__(self):
        return len(self.arrays["targets"])

    def batch(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        a = self.arrays
        x = np.empty((len(indices), 19, 8, 8), dtype=np.float32)
        x[:, :18] = a["boards"][indices]
        x[:, 18] = a["clocks"][indices, None, None] / 100.0
        mask = np.zeros((len(indices), ACTIONS), dtype=bool)
        for row, i in enumerate(indices):
            mask[row, a["legal"][a["offsets"][i]:a["offsets"][i + 1]]] = True
        return (torch.from_numpy(x).to(device), torch.from_numpy(a["targets"][indices].copy()).to(device),
                torch.from_numpy(mask).to(device), a["plies"][indices].copy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparar cache compacto para treinamento")
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/levon_aronian")
    parser.add_argument("--output", type=Path, default=ROOT / "data/cache/aronian_v1")
    parser.add_argument("--limit", type=int, help="Somente diagnóstico: limitar cada divisão")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit deve ser positivo")
    prepare(args.source, args.output, args.limit)
