"""Local chess UI backed by the trained policy network. No extra dependencies."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import urlparse
import webbrowser

import chess

from .common import ROOT
from .predict import choose_move, load_model, reconstruct

STATIC = ROOT / "web/dist"


def position_response(board, human_color):
    outcome = board.outcome(claim_draw=True)
    finished = outcome is not None
    if outcome:
        if outcome.winner is None:
            reasons = {"STALEMATE": "afogamento", "INSUFFICIENT_MATERIAL": "material insuficiente",
                       "THREEFOLD_REPETITION": "repetição", "FIVEFOLD_REPETITION": "repetição",
                       "FIFTY_MOVES": "regra dos 50 lances", "SEVENTYFIVE_MOVES": "regra dos 75 lances"}
            status = "Empate por " + reasons.get(outcome.termination.name, "regra da partida") + "."
        else:
            status = "Você venceu por xeque-mate." if outcome.winner == human_color else "O modelo venceu por xeque-mate."
    else:
        status = "Sua vez" if board.turn == human_color else "O modelo está pensando"
        status += " — xeque." if board.is_check() else "."
    return {
        "board": {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()},
        "fen": board.fen(en_passant="fen"), "moves": [m.uci() for m in board.move_stack],
        "turn": "white" if board.turn else "black", "humanColor": "white" if human_color else "black",
        "legalMoves": [m.uci() for m in board.legal_moves] if not finished and board.turn == human_color else [],
        "lastMove": board.peek().uci() if board.move_stack else None,
        "checkSquare": chess.square_name(board.king(board.turn)) if board.is_check() else None,
        "finished": finished, "status": status,
        "result": outcome.result() if outcome else None,
    }


class GameService:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()

    def play(self, payload):
        if not isinstance(payload, dict) or payload.get("color") not in ("white", "black"):
            raise ValueError("Escolha brancas ou pretas.")
        moves = payload.get("moves", [])
        if not isinstance(moves, list) or len(moves) > 2048 or any(not isinstance(m, str) or len(m) not in (4, 5) for m in moves):
            raise ValueError("Histórico de jogadas inválido.")
        board = reconstruct(moves=moves)
        human = payload["color"] == "white"
        if not board.is_game_over(claim_draw=True) and board.turn != human:
            with self.lock:
                decision = choose_move(self.model, board)
            if decision["move_uci"]:
                board.push_uci(decision["move_uci"])
        return position_response(board, human)


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, service, **kwargs):
        self.service = service
        super().__init__(*args, **kwargs)

    def send_bytes(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status, value):
        self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def allowed_host(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        return self.headers.get("Host") in hosts

    def do_GET(self):
        if not self.allowed_host():
            self.send_json(403, {"error": "Acesso apenas pelo endereço local."})
            return
        route = urlparse(self.path).path
        if route == "/api/health":
            self.send_json(200, {"ready": True})
            return
        relative = "index.html" if route == "/" else route.lstrip("/")
        path = (STATIC / relative).resolve()
        if not path.is_relative_to(STATIC.resolve()) or not path.is_file():
            self.send_json(404, {"error": "Não encontrado."})
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            mime = "text/javascript"
        self.send_bytes(200, path.read_bytes(), mime + ("; charset=utf-8" if mime.startswith("text/") else ""))

    def do_POST(self):
        origin = self.headers.get("Origin")
        allowed_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if not self.allowed_host() or (origin and origin not in allowed_origins):
            self.send_json(403, {"error": "Origem inválida."})
            return
        if urlparse(self.path).path != "/api/play":
            self.send_json(404, {"error": "Não encontrado."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Requisição inválida.")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Formato esperado: JSON.")
            payload = json.loads(self.rfile.read(length))
            result = self.service.play(payload)
        except (ValueError, TypeError, KeyError) as error:
            self.send_json(400, {"error": str(error)})
            return
        except Exception:
            self.log_error("Erro durante a inferência")
            self.send_json(500, {"error": "Não foi possível calcular a jogada. Tente novamente."})
            return
        self.send_json(200, result)

    def log_message(self, format, *args):
        if args and str(args[1] if len(args) > 1 else "") not in ("200", "304"):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Jogar contra o modelo no navegador, localmente")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/aronian_v1/best_weights.pt")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Abrir o navegador automaticamente")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Pesos não encontrados: {args.weights}")
    print("Carregando o modelo em CPU...", flush=True)
    model, bundle = load_model(args.weights)
    if bundle["diagnostic_only"]:
        print("Aviso: modelo de diagnóstico.", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, service=GameService(model)))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Jogue em {url} | modelo da época {bundle['epoch']}", flush=True)
    print("Ctrl+C encerra o servidor.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
