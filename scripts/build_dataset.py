"""Download public PGNs and extract only Levon Aronian's decisions."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://www.pgnmentor.com/players/Aronian.zip"
PLAYER = "levon_aronian"
ALIASES = {"aronianlevon", "levonaronian", "aronianl"}


def normalized(name):
    return re.sub(r"[^a-z]", "", name.casefold())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_jsonl(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def split_for(sequence_id):
    # Identical full move sequences always stay together, including different events.
    bucket = int(hashlib.sha256(("aronian-v1:" + sequence_id).encode()).hexdigest()[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def normalize_pgn(text):
    text = text.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    # A source export has two blank lines between headers and move 1, which
    # python-chess otherwise interprets as an empty game plus an anonymous game.
    return re.subn(r"(?m)^(\[[^\n]*\])\n(?:[ \t]*\n){2,}(?=[ \t]*\d+\.)", r"\1\n\n", text)


def download(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "Aronian.zip"
    if not archive.exists():
        req = urllib.request.Request(SOURCE, headers={"User-Agent": "ChessImitationDataset/1.0"})
        with urllib.request.urlopen(req, timeout=90) as response:
            payload = response.read()
        # Validate before caching. Never extract arbitrary archive paths.
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            z.read("Aronian.pgn")
        archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as z:
        pgn = z.read("Aronian.pgn")
    path = raw_dir / "Aronian.pgn"
    path.write_bytes(pgn)
    manifest_path = raw_dir / "source.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "url": SOURCE,
        "catalog_url": "https://www.pgnmentor.com/files.html",
        "first_recorded_at_utc": previous.get("first_recorded_at_utc") if previous.get("zip_sha256") == sha else datetime.now(timezone.utc).isoformat(),
        "zip_sha256": sha,
        "pgn_sha256": hashlib.sha256(pgn).hexdigest(),
        "pgn_bytes": len(pgn),
        "availability": "Public free download; no explicit dataset redistribution license recorded.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, manifest


def prepare_game(game):
    """Validate the WHOLE game before returning any training examples."""
    if game.errors:
        raise ValueError("parse_error")
    if game.headers.get("Variant", "Standard").lower() not in {"standard", "chess", "normal"}:
        raise ValueError("unsupported_variant")
    matches = [color for color, tag in ((chess.WHITE, "White"), (chess.BLACK, "Black"))
               if normalized(game.headers.get(tag, "")) in ALIASES]
    if len(matches) != 1:
        raise ValueError("target_identity_missing_or_ambiguous")
    color = matches[0]
    board = game.board()
    if not board.is_valid():
        raise ValueError("invalid_start_position")
    initial_fen = board.fen(en_passant="fen")
    moves = list(game.mainline_moves())
    if not moves:
        raise ValueError("no_moves")
    examples = []
    for ply, move in enumerate(moves, start=1):
        if not move or not board.is_legal(move):
            raise ValueError("illegal_or_null_move")
        if board.turn == color:
            examples.append({
                "player_id": PLAYER,
                "color": "white" if color else "black",
                "ply": ply,
                "fullmove_number": board.fullmove_number,
                "fen": board.fen(en_passant="fen"),
                "move_uci": move.uci(),
                "move_san": board.san(move),
            })
        board.push(move)
    if not examples:
        raise ValueError("no_target_moves")
    uci = [m.uci() for m in moves]
    sequence_id = digest([initial_fen, uci])
    # Repeated exports with changed annotations/Elo/event spelling are duplicates.
    identity = [normalized(game.headers.get(tag, "")) for tag in ("White", "Black")]
    identity[0 if color else 1] = PLAYER
    game_id = digest([identity, game.headers.get("Date", "?"), sequence_id])
    return {
        "game_id": game_id,
        "sequence_id": sequence_id,
        "split": split_for(sequence_id),
        "player_id": PLAYER,
        "target_color": "white" if color else "black",
        "initial_fen": initial_fen,
        "moves_uci": uci,
        "headers": dict(game.headers),
    }, examples


def build(pgn_path, output, source):
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").unlink(missing_ok=True)
    payload = pgn_path.read_bytes()
    try:
        text = payload.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = payload.decode("cp1252")
        encoding = "cp1252"
    text, header_spacing_repairs = normalize_pgn(text)
    stats = Counter()
    split_games, split_moves, colors, years, missing = (Counter() for _ in range(5))
    seen = set()
    sample = []
    with ExitStack() as stack:
        games_out = stack.enter_context(gzip.open(output / "games.jsonl.gz", "wt", encoding="utf-8"))
        rejects = stack.enter_context((output / "excluded.jsonl").open("w", encoding="utf-8"))
        outputs = {s: stack.enter_context(gzip.open(output / f"{s}.jsonl.gz", "wt", encoding="utf-8"))
                   for s in ("train", "validation", "test")}
        stream = io.StringIO(text)
        while (game := chess.pgn.read_game(stream)) is not None:
            stats["source_records"] += 1
            ordinal = stats["source_records"]
            try:
                record, examples = prepare_game(game)
                if record["game_id"] in seen:
                    raise ValueError("duplicate_game")
            except ValueError as error:
                reason = str(error)
                stats["excluded_records"] += 1
                stats[f"excluded_{reason}"] += 1
                write_jsonl(rejects, {"source_record": ordinal, "reason": reason, "headers": dict(game.headers),
                                     "parse_errors": [str(e) for e in game.errors]})
                continue
            seen.add(record["game_id"])
            record["source_record"] = ordinal
            write_jsonl(games_out, record)
            split = record["split"]
            split_games[split] += 1
            split_moves[split] += len(examples)
            colors[record["target_color"]] += len(examples)
            date = game.headers.get("Date", "")
            years[date[:4] if re.match(r"\d{4}", date) else "unknown"] += 1
            for tag in ("TimeControl", "WhiteElo", "BlackElo", "Date"):
                if game.headers.get(tag, "") in {"", "?", "-", "????.??.??"}:
                    missing[tag] += 1
            for example in examples:
                example.update(game_id=record["game_id"], split=split)
                write_jsonl(outputs[split], example)
                if len(sample) < 8:
                    sample.append(example)
            stats["accepted_games"] += 1
            stats["target_moves"] += len(examples)
            if ordinal % 500 == 0:
                print(f"Processed {ordinal} records; {stats['target_moves']} target moves", flush=True)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "player": "Levon Aronian", "accepted_normalized_aliases": sorted(ALIASES),
        "source": source, "encoding": encoding, "header_spacing_repairs": header_spacing_repairs, "counts": dict(stats),
        "games_by_split": dict(split_games), "moves_by_split": dict(split_moves),
        "moves_by_color": dict(colors), "games_by_year": dict(sorted(years.items())),
        "games_missing_metadata": dict(missing),
        "split_method": "SHA256 of full move-sequence group; approximately 80/10/10; seed aronian-v1",
        "limitations": [
            "Career games include childhood and all available time controls; not restricted to GM-title period.",
            "Different games can share positions/openings across splits; this is not a position-disjoint evaluation.",
            "FEN alone has no repetition history; reconstruct it using games.moves_uci[:ply-1].",
            "Source coverage is incomplete and source metadata may be inaccurate.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample:
            write_jsonl(handle, row)
    print(json.dumps({"counts": dict(stats), "moves_by_split": dict(split_moves)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, help="Use a local PGN instead of downloading")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/levon_aronian")
    args = parser.parse_args()
    if args.pgn:
        pgn_path = args.pgn
        source = {"local_path": str(pgn_path.resolve()), "pgn_sha256": hashlib.sha256(pgn_path.read_bytes()).hexdigest()}
    else:
        pgn_path, source = download(ROOT / "data/raw")
    build(pgn_path, args.output, source)


if __name__ == "__main__":
    main()
