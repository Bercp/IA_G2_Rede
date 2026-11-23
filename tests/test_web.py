import json
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import chess

from aronian.predict import reconstruct
from aronian.web import GameService, Handler, position_response


class WebTests(unittest.TestCase):
    def test_white_starts_without_model_move(self):
        with patch("aronian.web.choose_move") as choose:
            result = GameService(None).play({"color": "white", "moves": []})
            self.assertEqual(result["moves"], [])
            self.assertIn("e2e4", result["legalMoves"])
            choose.assert_not_called()

    def test_black_receives_model_opening(self):
        with patch("aronian.web.choose_move", return_value={"move_uci": "e2e4"}):
            result = GameService(None).play({"color": "black", "moves": []})
        self.assertEqual(result["moves"], ["e2e4"])
        self.assertEqual(result["board"]["e4"], "P")
        self.assertIn("e7e5", result["legalMoves"])

    def test_model_replies_to_human_move(self):
        with patch("aronian.web.choose_move", return_value={"move_uci": "e7e5"}):
            result = GameService(None).play({"color": "white", "moves": ["e2e4"]})
        self.assertEqual(result["moves"], ["e2e4", "e7e5"])
        self.assertEqual(result["turn"], "white")

    def test_invalid_history_is_rejected(self):
        for payload in ({"color": "white", "moves": ["e2e5"]}, {"color": "red"}, {"color": "white", "moves": "e2e4"}):
            with self.assertRaises(ValueError):
                GameService(None).play(payload)

    def test_special_moves_are_available(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.assertIn("e1g1", position_response(board, chess.WHITE)["legalMoves"])
        board = reconstruct(moves=["e2e4", "a7a6", "e4e5", "d7d5"])
        self.assertIn("e5d6", position_response(board, chess.WHITE)["legalMoves"])
        board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        moves = position_response(board, chess.WHITE)["legalMoves"]
        self.assertTrue(all("a7a8" + p in moves for p in "qrbn"))

    def test_finished_game_has_no_model_reply(self):
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
        with patch("aronian.web.choose_move") as choose:
            result = GameService(None).play({"color": "white", "moves": moves})
            choose.assert_not_called()
        self.assertTrue(result["finished"])
        self.assertEqual(result["result"], "0-1")
        self.assertEqual(result["legalMoves"], [])

    def test_http_static_and_post_contract(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, service=GameService(None)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(url) as response:
                self.assertIn(b'id="board"', response.read())
            request = Request(url + "/api/play", data=json.dumps({"color": "white", "moves": []}).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request) as response:
                self.assertEqual(len(json.load(response)["legalMoves"]), 20)
            for path in ("/runs/aronian_v1/best_weights.pt", "/../requirements.txt"):
                with self.assertRaises(HTTPError) as error:
                    urlopen(url + path)
                self.assertEqual(error.exception.code, 404)
            request = Request(url + "/api/play", data=b'{}', headers={"Content-Type": "application/json", "Origin": "https://example.com"})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
