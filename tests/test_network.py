import unittest

import chess
import numpy as np
import torch

from aronian.encoding import encode_board, id_to_move, move_to_id, legal_ids
from aronian.model import PolicyNet
from aronian.predict import reconstruct, choose_move


class NetworkTests(unittest.TestCase):
    def test_action_roundtrip_including_special_moves(self):
        for uci in ("e2e4", "e1g1", "e8c8", "e5d6", "a7a8q", "h2h1n", "a7b8b", "b2a1r"):
            self.assertEqual(id_to_move(move_to_id(chess.Move.from_uci(uci))).uci(), uci)

    def test_planes_and_orientation(self):
        board = reconstruct(moves=["e2e4"])
        x = encode_board(board)
        self.assertEqual(x.shape, (19, 8, 8))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x[0, 3, 4], 1)  # White pawn e4.
        self.assertEqual(x[3, 0, 0], 1)  # White rook a1.
        self.assertEqual(x[9, 7, 0], 1)  # Black rook a8.
        self.assertEqual(x[12].sum(), 0)
        self.assertEqual(x[17, 2, 4], 1)  # e3, FEN en passant target.
        self.assertEqual(x[13:17].sum(), 4 * 64)

    def test_output_order_matches_action_contract(self):
        net = PolicyNet(channels=4, blocks=1).eval()
        with torch.no_grad():
            net.head.weight.zero_()
            net.head.bias.copy_(torch.arange(320))
            logits = net(torch.zeros(1, 19, 8, 8))[0]
        for uci in ("a1h8", "h8a1", "a7a8q", "h2h1n"):
            move = chess.Move.from_uci(uci)
            self.assertEqual(float(logits[move_to_id(move)]), move_to_id(move) % 320)

    def test_legal_prediction_and_terminal_handling(self):
        net = PolicyNet(channels=4, blocks=1).eval()
        for board in (chess.Board(), reconstruct(moves=["e2e4"]),
                      reconstruct(moves=["e2e4", "a7a6", "e4e5", "d7d5"]),
                      chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")):
            result = choose_move(net, board)
            self.assertIn(chess.Move.from_uci(result["move_uci"]), board.legal_moves)
        mate = reconstruct(moves=["f2f3", "e7e5", "g2g4", "d8h4"])
        self.assertIsNone(choose_move(net, mate)["move_uci"])
        repetition = reconstruct(moves=["g1f3", "g8f6", "f3g1", "f6g8"] * 2)
        self.assertEqual(choose_move(net, repetition)["result"], "1/2-1/2")

    def test_reject_illegal_history(self):
        with self.assertRaises(ValueError):
            reconstruct(moves=["e2e5"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
