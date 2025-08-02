"""
A backend service for human to play with the best model. Use the index.html to open the web UI.
"""

import copy
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import best_model, board_size, current_model
from evaluate import sample_from_top_k_with_ties
from game import Board
from policy_value_net import PolicyValueNet
from mcts import MCTSPlayer

session_boards: Dict[str, Board] = {}
best_value_net = PolicyValueNet(board_size, model_file=best_model)
current_value_net = PolicyValueNet(board_size, model_file=current_model)
# choose the net that you are interested in playing against
value_net = best_value_net

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class MoveRequest(BaseModel):
    x: int
    y: int
    session_id: str

class ResetRequest(BaseModel):
    session_id: str

class FirstMoveRequest(BaseModel):
    session_id: str

def get_action(board):
    # legal_probs, value, legal_positions = value_net.policy_value_fn(board)
    # move = sample_from_top_k_with_ties(legal_positions, legal_probs, k=1)
    p = MCTSPlayer(value_net.policy_value_fn)
    return p.get_action(board, False)

@app.post("/get_first_move")
def get_first_move(data: FirstMoveRequest):
    session_id = data.session_id
    session_boards[session_id] = Board(board_size, start_player=-1)
    board = session_boards[session_id]
    move_x_y = [board_size//2, board_size//2]
    move = board.location_to_move(move_x_y)
    board.do_move(move)
    return {"move": move_x_y, "status": "ok", "winner": "unknown"}

@app.post("/get_move")
def get_move(data: MoveRequest):
    x, y, session_id = data.x, data.y, data.session_id
    if session_id not in session_boards:
        session_boards[session_id] = Board(board_size)
    board = session_boards[session_id]
    
    move = board.location_to_move([x, y])    
    if not board.is_legal_position(x, y):
        return {"move": [-1, -1], "status": "invalid_human_move", "winner":"unknown"}
    board.do_move(move)
    board.print_board()
    
    end, winner = board.game_end()
    if end:
        board = Board(board_size)
        if winner == 0:
            return {"move": [-1, -1], "status": "ok", "winner": "draw"}
        return {"move": [-1, -1], "status": "ok", "winner": "human"}
    
    # AI move
    move, _ = get_action(board)
    board.do_move(move)
    ai_r, ai_c = divmod(move, board.size) 
    print(f"Ai move: x={x}, y={y}", flush=True)
    board.print_board()
    end, winner = board.game_end()
    if end:
        board = Board(board_size)
        if winner == 0:
           return {"move": [-1, -1], "status": "ok", "winner": "draw"} 
        return {"move": [-1, -1], "status": "ok", "winner": "ai"}

    return {"move": [int(ai_r), int(ai_c)], "status": "ok", "winner": "unknown"}


@app.post("/reset")
async def reset_game(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    if session_id in session_boards:
        session_boards[session_id]=Board(board_size)
    print("new game start. reset.")
    return {"status": "reset"}


@app.post("/stop_session")
async def reset_game(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    if session_id in session_boards:
        del session_boards[session_id]
    print("page closed. reset.")
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)