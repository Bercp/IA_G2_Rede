"""Versioned representation shared by training and inference."""
import chess
import numpy as np

VERSION = "aronian-19planes-20480-v1"
ACTIONS = 64 * 64 * 5
PROMOTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
SPEC = {
    "version": VERSION, "input_shape": [None, 19, 8, 8],
    "square_order": "a1=0, b1=1, ..., h8=63; row=rank-1; column=file",
    "planes": ["white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
               "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
               "white_to_move", "white_kingside", "white_queenside", "black_kingside", "black_queenside",
               "en_passant_square", "min(halfmove_clock,100)/100"],
    "action_formula": "((from_square*64+to_square)*5)+promotion_code",
    "promotions": [None, "n", "b", "r", "q"], "output_size": ACTIONS,
    "output": "raw logits; apply legal mask outside model",
}


def move_to_id(move):
    if not move or move.drop:
        raise ValueError("Null moves and variants are unsupported")
    return (move.from_square * 64 + move.to_square) * 5 + PROMOTIONS.index(move.promotion)


def id_to_move(index):
    if not 0 <= int(index) < ACTIONS:
        raise ValueError("Action index out of range")
    squares, promotion = divmod(int(index), 5)
    origin, target = divmod(squares, 64)
    return chess.Move(origin, target, promotion=PROMOTIONS[promotion])


def compact_board(board):
    planes = np.zeros((18, 8, 8), dtype=np.uint8)
    for square, piece in board.piece_map().items():
        channel = piece.piece_type - 1 + (0 if piece.color else 6)
        planes[channel, square // 8, square % 8] = 1
    flags = (board.turn, board.has_kingside_castling_rights(chess.WHITE),
             board.has_queenside_castling_rights(chess.WHITE),
             board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK))
    for channel, flag in enumerate(flags, 12):
        planes[channel].fill(int(flag))
    if board.ep_square is not None:
        planes[17, board.ep_square // 8, board.ep_square % 8] = 1
    return planes, min(board.halfmove_clock, 100)


def encode_board(board):
    planes, clock = compact_board(board)
    result = np.empty((19, 8, 8), dtype=np.float32)
    result[:18] = planes
    result[18].fill(clock / 100)
    return result


def legal_ids(board):
    return np.array([move_to_id(m) for m in board.legal_moves], dtype=np.int64)
