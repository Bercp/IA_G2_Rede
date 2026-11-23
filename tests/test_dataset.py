import io
import unittest
import chess.pgn
from scripts.build_dataset import prepare_game, normalize_pgn


def parse(text):
    return chess.pgn.read_game(io.StringIO(text))


class DatasetTests(unittest.TestCase):
    def test_extra_blank_line_does_not_detach_moves_from_headers(self):
        text, repairs = normalize_pgn('[White "Aronian,L"]\n\n\n1. e4 e5 *\n\n[Black "Aronian,L"]\n\n1. d4 d5 *')
        stream = io.StringIO(text)
        self.assertEqual(repairs, 1)
        self.assertEqual(prepare_game(chess.pgn.read_game(stream))[1][0]["move_uci"], "e2e4")
        self.assertEqual(prepare_game(chess.pgn.read_game(stream))[1][0]["move_uci"], "d7d5")
        self.assertIsNone(chess.pgn.read_game(stream))

    def test_white_and_black_targets_and_history(self):
        for tag, expected in (("White", ["e2e4", "g1f3"]), ("Black", ["e7e5", "b8c6"])):
            game = parse(f'[{tag} "Aronian, Levon"]\n\n1. e4 e5 2. Nf3 Nc6 *')
            record, rows = prepare_game(game)
            self.assertEqual([r["move_uci"] for r in rows], expected)
            self.assertEqual(record["moves_uci"], ["e2e4", "e7e5", "g1f3", "b8c6"])
            for row in rows:
                board = chess.Board(record["initial_fen"])
                for uci in record["moves_uci"][:row["ply"] - 1]:
                    board.push_uci(uci)
                self.assertEqual(board.fen(en_passant="fen"), row["fen"])

    def test_excludes_other_aronian(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            prepare_game(parse('[White "Aronian,Laurel"]\n\n1. e4 e5 *'))

    def test_alias_and_annotation_deduplication(self):
        a, _ = prepare_game(parse('[White "Aronian, Levon"]\n\n1. e4 e5 *'))
        b, _ = prepare_game(parse('[White "Aronian,L"]\n\n1. e4 {note} e5 *'))
        self.assertEqual(a["game_id"], b["game_id"])

    def test_same_sequence_different_dates_stays_in_same_split(self):
        records = [prepare_game(parse(f'[White "Aronian,L"]\n[Date "{year}.01.01"]\n\n1. e4 e5 *'))[0]
                   for year in (2020, 2021)]
        self.assertNotEqual(records[0]["game_id"], records[1]["game_id"])
        self.assertEqual(records[0]["split"], records[1]["split"])

    def test_promotion_from_custom_position(self):
        game = parse('[White "Aronian,L"]\n[SetUp "1"]\n[FEN "7k/P7/8/8/8/8/8/7K w - - 0 1"]\n\n1. a8=Q+ *')
        _, rows = prepare_game(game)
        self.assertEqual(rows[0]["move_uci"], "a7a8q")

    def test_whole_game_rejected_after_illegal_move(self):
        game = parse('[White "Aronian,L"]\n\n1. e4 e5 2. Bh6 *')
        with self.assertRaisesRegex(ValueError, "parse_error"):
            prepare_game(game)

    def test_null_move_is_not_a_training_label(self):
        game = parse('[White "Aronian,L"]\n\n1. e4 -- *')
        with self.assertRaisesRegex(ValueError, "null_move"):
            prepare_game(game)


if __name__ == "__main__":
    unittest.main()
