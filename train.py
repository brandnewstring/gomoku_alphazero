import gc
import os
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow.keras.backend as K
from pympler import asizeof

import evaluate
from config import best_model, board_size, current_model, draw_threshold
from game import Board
from mcts import MCTSPlayer
from policy_value_net import PolicyValueNet

# training params
temp = 1.0  # the temperature param
learn_rate = 2e-3
lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
batch_size = 512  # mini-batch size for training
data_buffer = deque(maxlen=5000) # data buffer with size 10000
epochs = 5  # num of train_steps for each update
kl_targ = 0.02
game_batch_num = 1000

def train():
    start_time = time.time()
    for i in range(game_batch_num):
        new_play_data = None
        # avoid memory not being released
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(collect_selfplay_data)
            new_play_data = future.result()

        data_buffer.extend(new_play_data)
        print(
            f"\033[33m"
            f"batch i: {i+1}, "
            f"new data len: {len(new_play_data)}, "
            f"data buffer len: {len(data_buffer)}, "
            f"data buffer size: {asizeof.asizeof(data_buffer)/(1024*1024):.2f} MB"
            f"\033[0m")

        if len(data_buffer) > batch_size:
            loss, entropy = policy_update()

        time_used = time.time() - start_time
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(
            f"\033[33m"
            f"[{current_time}] ⏱️ Used: {int(time_used // 3600)}h "
            f"{int((time_used % 3600) // 60)}min "
            f"{int(time_used % 60)}s"
            f"\033[0m"
        )
        
        gc.collect()
        K.clear_session()

        if (len(data_buffer) > batch_size):
            # evaluate and save the best model form time to time, which is async.
            evaluate.run(i)


def start_self_play(board, mcts_player, is_shown=0, temp=1e-3):
    """ start a self-play game using a MCTS player, reuse the search tree,
    and store the self-play data: (state, mcts_probs, z) for training
    """
    states, mcts_probs, current_players = [], [], []
    while True:
        move, move_probs = mcts_player.get_action(board, temp=temp)
        # store the data
        states.append(board.current_state().astype(np.float32))
        mcts_probs.append(move_probs.astype(np.float32))
        current_players.append(board.current_player)
        board.do_move(move)
        end, winner = board.game_end()
        if end:
            # winner from the perspective of the current player of each state
            winners_z = np.zeros(len(current_players))
            if winner != 0:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0

            if is_shown:
                board.print_board()
                if winner != 0:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
            
            mcts_player.reset_player()
            return winner, zip(states, mcts_probs, winners_z)
        
def collect_selfplay_data():
    """collect self-play data for training. returns the new data"""
    policy_value_net = None
    if os.path.exists(current_model):
        policy_value_net = PolicyValueNet(board_size, model_file=current_model)
    else:
        policy_value_net = PolicyValueNet(board_size)

    # create one board per game
    board = Board(board_size, draw_threshold=draw_threshold)
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn)
    winner, play_data = start_self_play(board, mcts_player, temp=temp)
    play_data = list(play_data)[:]
    play_data = augment_data_by_symmetry(play_data)

    policy_value_net.save_model(current_model)
    return play_data


def augment_data_by_symmetry(play_data):
    """Augment self-play data using board symmetries (rotating 90°, 180°, 270°)."""
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(
                np.flipud(mcts_prob.reshape(board_size, board_size)), i)
            extend_data.append(
                (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append(
                (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
    return extend_data
                        
def policy_update():
    """update the policy-value net"""
    global lr_multiplier

    policy_value_net = PolicyValueNet(board_size, model_file=current_model)

    mini_batch = random.sample(data_buffer, batch_size)
    state_batch = np.array([data[0] for data in mini_batch],
                           dtype=np.float32)
    mcts_probs_batch = np.array([data[1] for data in mini_batch],
                                dtype=np.float32)
    winner_batch = np.array([data[2] for data in mini_batch],
                            dtype=np.float32)

    state_batch = state_batch.reshape(-1, 4, board_size, board_size)
    mcts_probs_batch = mcts_probs_batch.reshape(-1,
                                                board_size * board_size)
    winner_batch = winner_batch.reshape(-1, 1)

    old_probs, old_v = policy_value_net.policy_value(state_batch)
    for i in range(epochs):
        loss, entropy = policy_value_net.train_step(
            state_batch,
            mcts_probs_batch,
            winner_batch,
            learn_rate * lr_multiplier,
        )
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(
            np.sum(
                old_probs *
                (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1,
            ))
        if kl > kl_targ * 4:  # early stopping if D_KL diverges badly
            break
    # adaptively adjust the learning rate
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    # when first game is draw, var can be 0 and needs to be handled
    var_y = np.var(np.array(winner_batch))
    if var_y < 1e-8:
        explained_var_old = np.nan
        explained_var_new = np.nan
    else:
        explained_var_old = 1 - np.var(
            np.array(winner_batch) - old_v.flatten()) / np.var(
                np.array(winner_batch))
        explained_var_new = 1 - np.var(
            np.array(winner_batch) - new_v.flatten()) / np.var(
                np.array(winner_batch))

    # always save the new model
    policy_value_net.save_model(current_model)
    print(f"\033[33m"
          f"kl:{kl:.5f}, "
          f"lr_multiplier:{lr_multiplier:.3f}, "
          f"loss:{loss}, "
          f"entropy:{entropy}, "
          f"explained_var_old:{explained_var_old:.3f}, "
          f"explained_var_new:{explained_var_new:.3f}"
          f"\033[0m")
    return loss, entropy


if __name__ == "__main__":
    train()
