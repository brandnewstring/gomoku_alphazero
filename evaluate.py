import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from config import best_model, board_size, current_model, draw_threshold
from game import Board
from policy_value_net import PolicyValueNet

# params for model evaluation
check_freq = 15 # how frequent we check for the best model
total_games = 120 # total num of games we run to determine which model is better
win_threshold = 0.02 # the win rate threshold for a model to be considered better

def sample_from_top_k_with_ties(input_dict, k):
    """
    从概率前k（含并列）的元素中按归一化概率随机返回一个key。

    参数:
        input_dict (dict): 形如 {元素: 概率} 的字典。
        k (int): 前k名（含并列）

    返回:
        key: 按归一化概率选中的一个key

    异常:
        ValueError: 如果选中的前k项概率总和为0，则抛出异常。
    """
    if isinstance(input_dict, zip):
        input_dict = dict(input_dict)
    
    if not input_dict or k <= 0:
        raise ValueError("输入字典不能为空，且k必须大于0。")

    # 步骤1：获取前k名的概率阈值
    sorted_probs = sorted(set(input_dict.values()), reverse=True)
    if len(sorted_probs) < k:
        threshold = sorted_probs[-1]
    else:
        threshold = sorted_probs[k - 1]

    # 步骤2：获取所有满足条件的key及其原始概率
    top_items = {key: prob for key, prob in input_dict.items() if prob >= threshold}

    # 步骤3：归一化概率
    total = sum(top_items.values())
    if total == 0:
        raise ValueError("前k名（含并列）的元素总概率为0，无法归一化。")

    probs = [v / total for v in top_items.values()]
    keys = list(top_items.keys())

    # 步骤4：按归一化概率抽样
    return np.random.choice(keys, p=probs)

def start_evaluate(board, value_net_1, value_net_2, is_shown=0):
    """evaluates and save the best model from time to time"""
    players = {1: value_net_1, -1: value_net_2}
    while True:
        player_in_turn = players[board.current_player]
        act_probs, value = player_in_turn.policy_value_fn(board)
        # when evaluating, randomly select in the best 3 possible moves
        move = sample_from_top_k_with_ties(act_probs, 3)
        board.do_move(move)
        
        end, winner = board.game_end()
        if end:
            if is_shown:
                board.print_board()
            return winner

def policy_evaluate_using_cpu():
    """Removes CUDA_VISIBLE_DEVICES so that CPU is used. Must be used in multiprocessing to avoid env issues."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    policy_evaluate()
        

def policy_evaluate():
    """
    Compare the models and save the best model.
    """
    # load the current and the best model
    if not os.path.exists(best_model):
        current_value_net = PolicyValueNet(board_size, model_file=current_model)
        current_value_net.save_model(best_model)
        return
    
    best_value_net = PolicyValueNet(board_size, model_file=best_model)
    current_value_net = PolicyValueNet(board_size, model_file=current_model)

    # run the games and check the win rate
    win_cnt = Counter()
    start_player = 1
    for i in range(total_games):
        board = Board(board_size, start_player=start_player, draw_threshold=draw_threshold)
        winner = start_evaluate(board, current_value_net, best_value_net)
        win_cnt[winner] += 1
        start_player = -start_player

    diff = (win_cnt[1] - win_cnt[-1]) / total_games
    if diff > win_threshold:
        current_value_net.save_model(best_model)
        print(f"\033[31m"
          f"total_games: {total_games}, "
          f"win: {win_cnt[1]}, lose: {win_cnt[-1]}, tie: {win_cnt[0]}"
          f"\nBest model found and saved."
          f"\033[0m")
    else:
        print(f"\033[31m"
          f"total_games: {total_games}, "
          f"win: {win_cnt[1]}, lose: {win_cnt[-1]}, tie: {win_cnt[0]}"
          f"\nModel not saved."
          f"\033[0m")

def run(game_batch_count):
    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     future = executor.submit(policy_evaluate)
    #     result = future.result()
    if (game_batch_count + 1) % check_freq != 0:
        return
    
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.submit(policy_evaluate_using_cpu)
    
