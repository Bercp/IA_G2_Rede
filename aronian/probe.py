"""Record a small, descriptive tactical probe without changing trained weights."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch

from .common import ROOT, save_json, sha256
from .encoding import encode_board, id_to_move, legal_ids, move_to_id
from .predict import load_model, reconstruct


CASES = [
    {
        "name": "white_queen_on_h4",
        "sequence_san": "1. e4 e5 2. Qh5 Nc6 3. Qh4",
        "moves_uci": ["e2e4", "e7e5", "d1h5", "b8c6", "h5h4"],
        "examined_capture_uci": "d8h4",
    },
    {
        "name": "black_queen_on_h4",
        "sequence_san": "1. e4 e5 2. Nf3 Qh4",
        "moves_uci": ["e2e4", "e7e5", "g1f3", "d8h4"],
        "examined_capture_uci": "f3h4",
    },
    {
        "name": "white_queen_on_e5_with_check",
        "sequence_san": "1. e4 e5 2. Qh5 Nc6 3. Qxe5+",
        "moves_uci": ["e2e4", "e7e5", "d1h5", "b8c6", "h5e5"],
        "examined_capture_uci": "c6e5",
    },
]


@torch.inference_mode()
def probe(weights):
    model, bundle = load_model(weights)
    cases = []
    for case in CASES:
        board = reconstruct(moves=case["moves_uci"])
        ids = torch.from_numpy(legal_ids(board))
        logits = model(torch.from_numpy(encode_board(board)).unsqueeze(0))[0, ids]
        probabilities = logits.softmax(0)
        order = logits.argsort(descending=True)
        capture = chess.Move.from_uci(case["examined_capture_uci"])
        if capture not in board.legal_moves:
            raise ValueError("Examined capture must be legal")
        capture_id = move_to_id(capture)
        capture_index = int((ids == capture_id).nonzero()[0, 0])
        capture_rank = int((ids[order] == capture_id).nonzero()[0, 0]) + 1
        ranked = [{"move_uci": id_to_move(int(ids[i])).uci(),
                   "move_san": board.san(id_to_move(int(ids[i]))),
                   "probability": float(probabilities[i])} for i in order[:5]]
        cases.append({**case, "fen": board.fen(en_passant="fen"), "in_check": board.is_check(),
                      "examined_capture_san": board.san(capture), "examined_capture_legal": True,
                      "examined_capture_rank": capture_rank,
                      "examined_capture_probability": float(probabilities[capture_index]),
                      "chosen_move_uci": ranked[0]["move_uci"], "top5": ranked})
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "weights_sha256": sha256(weights),
            "model_epoch": bundle["epoch"], "diagnostic_only": bundle["diagnostic_only"],
            "method": "Greedy policy inference, temperature=0, legal-move filtering, CPU",
            "limitations": "Three descriptive examples; no statistical tactical accuracy or engine evaluation.",
            "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/aronian_v1/best_weights.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/aronian_v1/tactical_probes.json")
    args = parser.parse_args()
    result = probe(args.weights)
    save_json(args.output, result)
    for case in result["cases"]:
        print(case["sequence_san"], "->", case["top5"][0]["move_san"],
              "| capture rank:", case["examined_capture_rank"])
    print("Saved:", args.output)
