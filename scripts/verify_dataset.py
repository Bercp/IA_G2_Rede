"""Replay every exported game and verify every decision against its history."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import chess

from build_dataset import ALIASES, ROOT, normalized


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(folder):
    games = {}
    sequences = {}
    expected_total = 0
    for game in rows(folder / "games.jsonl.gz"):
        gid = game["game_id"]
        require(gid not in games, f"Duplicate game {gid}")
        games[gid] = game
        seq = game["sequence_id"]
        require(sequences.setdefault(seq, game["split"]) == game["split"], "Sequence crosses splits")
    counts = Counter()
    colors = Counter()
    seen = set()
    last_id, expected = None, {}
    for split in ("train", "validation", "test"):
        for row in rows(folder / f"{split}.jsonl.gz"):
            gid = row["game_id"]
            game = games[gid]
            require(row["split"] == game["split"] == split, "Game crosses splits")
            require(row["player_id"] == game["player_id"] == "levon_aronian", "Wrong player")
            require(row["color"] == game["target_color"], "Wrong target color")
            key = (gid, row["ply"])
            require(key not in seen, "Duplicate decision")
            seen.add(key)
            if gid != last_id:
                expected = {}
                board = chess.Board(game["initial_fen"])
                target = game["target_color"] == "white"
                tag = "White" if target else "Black"
                require(normalized(game["headers"][tag]) in ALIASES, "Wrong player identity")
                for ply, uci in enumerate(game["moves_uci"], 1):
                    move = chess.Move.from_uci(uci)
                    require(board.is_legal(move), "Illegal game move")
                    if board.turn == target:
                        expected[ply] = (board.fen(en_passant="fen"), uci, board.san(move), board.fullmove_number)
                    board.push(move)
                last_id = gid
            require(expected.get(row["ply"]) == (row["fen"], row["move_uci"], row["move_san"], row["fullmove_number"]), "Decision does not match game history")
            counts[split] += 1
            colors[row["color"]] += 1
            if sum(counts.values()) % 50000 == 0:
                print(f"Verified {sum(counts.values())} decisions", flush=True)
    # Completeness, including games accidentally missing entirely from decision files.
    for gid, game in games.items():
        starts_white = chess.Board(game["initial_fen"]).turn
        target_white = game["target_color"] == "white"
        for ply in range(1, len(game["moves_uci"]) + 1):
            white_to_move = starts_white if ply % 2 else not starts_white
            if white_to_move == target_white:
                expected_total += 1
                require((gid, ply) in seen, "Missing target move")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    require(dict(counts) == report["moves_by_split"], "Incorrect split report")
    require(dict(colors) == report["moves_by_color"], "Incorrect color report")
    require(len(games) == report["counts"]["accepted_games"], "Incorrect game count")
    require(expected_total == report["counts"]["target_moves"], "Incorrect move count")
    result = {"status": "passed", "games_replayed": len(games), "moves_verified": expected_total,
              "moves_by_split": dict(counts), "game_or_full_sequence_overlap_between_splits": 0}
    (folder / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/levon_aronian")
    verify(parser.parse_args().data)
