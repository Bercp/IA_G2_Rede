"""Inference with legal move filtering and full-game rule handling."""
import argparse
import json
from pathlib import Path

import chess
import torch

from .encoding import VERSION, encode_board, id_to_move, legal_ids
from .model import PolicyNet


def load_model(path, threads=4):
    torch.set_num_threads(threads)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["encoding"] != VERSION:
        raise ValueError("Unsupported encoding")
    model = PolicyNet(**bundle["architecture"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def reconstruct(initial_fen=chess.STARTING_FEN, moves=()):
    board = chess.Board(initial_fen)
    if not board.is_valid():
        raise ValueError("Invalid initial FEN")
    for uci in moves:
        if board.is_game_over():
            raise ValueError("Move supplied after game termination")
        board.push_uci(uci)
    return board


@torch.inference_mode()
def choose_move(model, board, temperature=0.0):
    if temperature < 0:
        raise ValueError("Temperature must be nonnegative")
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return {"move_uci": None, "result": outcome.result(), "termination": outcome.termination.name}
    ids = torch.from_numpy(legal_ids(board))
    logits = model(torch.from_numpy(encode_board(board)).unsqueeze(0))[0, ids]
    probabilities = torch.softmax(logits / (temperature if temperature > 0 else 1.0), dim=0)
    index = int(torch.multinomial(probabilities, 1)) if temperature > 0 else int(logits.argmax())
    move = id_to_move(int(ids[index]))
    return {"move_uci": move.uci(), "move_san": board.san(move), "probability": float(probabilities[index])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="Posição inicial do histórico fornecido")
    parser.add_argument("--moves", nargs="*", default=[])
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--play", action="store_true", help="Jogar de brancas no terminal; informar lances UCI")
    args = parser.parse_args()
    model, bundle = load_model(args.weights)
    if bundle["diagnostic_only"]:
        print("AVISO: pesos de diagnóstico, não representam um treinamento completo.")
    board = reconstruct(args.fen, args.moves)
    if args.play:
        while not board.is_game_over(claim_draw=True):
            print(board)
            if board.turn:
                uci = input("Seu lance UCI (ex.: e2e4) ou sair: ").strip()
                if uci.lower() == "sair":
                    break
                try:
                    board.push_uci(uci)
                except ValueError as error:
                    print(error)
            else:
                result = choose_move(model, board, args.temperature)
                print("Modelo:", result)
                board.push_uci(result["move_uci"])
        print("Resultado:", board.result(claim_draw=True))
    else:
        print(json.dumps(choose_move(model, board, args.temperature), indent=2))
