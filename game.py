import copy

import numpy as np


class Board(object):
    """The board for a single game."""

    def __init__(self, size, n_in_row=5, start_player=1, draw_threshold=1 ,**kwargs):
        if start_player not in (1, -1):
            raise ValueError("start_player must be 1 or -1")
        if n_in_row != 5:
            raise ValueError("n_in_row must be 5 for now due to implementation of get_valid_moves()")
        self.n_in_row = n_in_row
        self.start_player = start_player
        self.size = size
        self.draw_threshold = draw_threshold
        self.num_grids = size*size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.players = [1, -1]  # player1 and player2
        self.current_player = start_player  # start player
        self.total_moves = 0
        self.last_move = -1

    def is_legal_position(self, r, c):
        return self.within_boundary(r,c) and self.board[r][c]==0
    
    def get_valid_moves(self):
        """A slight complicated logic to get the next possible moves. This is specific for freestyle Gomoku. Renju and other rules are not implemented so far."""
        # if board empty, the first move is in the center
        if self.total_moves == 0:
            return [self.size // 2 * self.size + self.size // 2]
        if self.total_moves == self.num_grids:
            return []

        previous_player = self.current_player * -1
        
        #如果自己有4连且有地方走，直接5连
        for row in range(self.size):
            for col in range(self.size):
                if row < self.size - 4:
                    is_4_match, position = self.find_4_match(row, col, self.current_player, [1, 0])
                    if is_4_match == True: return position
                if col < self.size - 4:
                    is_4_match, position = self.find_4_match(row, col, self.current_player, [0, 1])
                    if is_4_match == True: return position
                if row < self.size - 4 and col < self.size - 4:
                    is_4_match, position = self.find_4_match(row, col, self.current_player, [1, 1])
                    if is_4_match == True: return position
                if row >= 4 and col < self.size - 4:
                    is_4_match, position = self.find_4_match(
                        row, col, self.current_player, [-1, 1])
                    if is_4_match == True: return position
        
        #如果对方有4连则必须堵空的地方
        for row in range(self.size):
            for col in range(self.size):                
                if row < self.size - 4:
                    is_4_match, position = self.find_4_match(row, col, previous_player, [1, 0])
                    if is_4_match == True: return position
                if col < self.size - 4:
                    is_4_match, position = self.find_4_match(row, col, previous_player, [0, 1])
                    if is_4_match == True: return position
                if row < self.size - 4 and col < self.size - 4:
                    is_4_match, position = self.find_4_match(row, col, previous_player, [1, 1])
                    if is_4_match == True: return position
                if row >= 4 and col < self.size - 4:
                    is_4_match, position = self.find_4_match(row, col, previous_player, [-1, 1])
                    if is_4_match == True: return position
        
        #如果自己有3连且两边没堵，则4连，这里返回两个点的情况有可能会有特殊情况不成立，在Renju规则下估计漏洞很多，以后可以改
        for row in range(self.size):
            for col in range(self.size):                  
                if row < self.size - 4:
                    is_3_match, position = self.find_3_match(row, col, self.current_player, [1, 0])
                    if is_3_match == True: return position
                if col < self.size - 4:
                    is_3_match, position = self.find_3_match(row, col, self.current_player, [0, 1])
                    if is_3_match == True: return position
                if row < self.size - 4 and col < self.size - 4:
                    is_3_match, position = self.find_3_match(row, col, self.current_player, [1, 1])
                    if is_3_match == True: return position
                if row >= 4 and col < self.size - 4:
                    is_3_match, position = self.find_3_match(row, col, self.current_player, [-1, 1])
                    if is_3_match == True: return position
        
        #如果对方有3连且没堵，堵住
        for row in range(self.size):
            for col in range(self.size):                  
                if row < self.size - 4:
                    is_3_match, position = self.find_3_match(row, col, previous_player, [1, 0])
                    if is_3_match == True: return position
                if col < self.size - 4:
                    is_3_match, position = self.find_3_match(row, col, previous_player, [0, 1])
                    if is_3_match == True: return position
                if row < self.size - 4 and col < self.size - 4:
                    is_3_match, position = self.find_3_match(row, col, previous_player, [1, 1])
                    if is_3_match == True: return position
                if row >= 4 and col < self.size - 4:
                    is_3_match, position = self.find_3_match(row, col, previous_player, [-1, 1])
                    if is_3_match == True: return position

        valid_moves = set()
        neighbors_range = [-2, -1, 0, 1, 2]
        # 只检测周围2步范围内有至少一个棋子的位置，为了提升效率，这里分成了2种实现
        # 当落子的数量小于空位的数量，则从有棋子的地方检测空位
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != 0:
                    # 周围2步内
                    for dr in neighbors_range:
                        for dc in neighbors_range:
                            new_row, new_col = row + dr, col + dc
                            # 如果这个位置在棋盘上，并且是空的，则加入到返回的集合中
                            if (0 <= new_row < self.size) and (
                                    0 <= new_col < self.size) and (
                                        self.board[new_row][new_col] == 0):
                                valid_moves.add(new_row * self.size + new_col)

        return list(valid_moves)

    def find_3_match(self, r, c, color, direction):
        dr = direction[0]
        dc = direction[1]
        # xx空x空
        if self.board[r][c] == color and self.board[r + dr][
                c + dc] == color and self.board[r + 2 * dr][
                    c + 2 * dc] == 0 and self.board[r + 3 * dr][
                        c + 3 * dc] == color and self.board[r + 4 * dr][
                            c + 4 * dc] == 0:
            if self.within_boundary(
                    r - dr, c - dc) and self.board[r - dr][c - dc] == 0:
                return True, [(r + 2 * dr) * self.size + c + 2 * dc]
            else:
                return True, [(r + 2 * dr) * self.size + c + 2 * dc,
                              (r + 4 * dr) * self.size + c + 4 * dc]
    # x空xx空
        if self.board[r][c] == color and self.board[r + dr][
                c + dc] == 0 and self.board[r + 2 * dr][
                    c + 2 * dc] == color and self.board[r + 3 * dr][
                        c + 3 * dc] == color and self.board[r + 4 * dr][
                            c + 4 * dc] == 0:
            if self.within_boundary(
                    r - dr, c - dc) and self.board[r - dr][c - dc] == 0:
                return True, [(r + dr) * self.size + c + dc]
            else:
                return True, [(r + dr) * self.size + c + dc,
                              (r + 4 * dr) * self.size + c + 4 * dc]

    # 空xxx空
        if self.board[r][c] == 0 and self.board[r + dr][
                c + dc] == color and self.board[r + 2 * dr][
                    c + 2 * dc] == color and self.board[r + 3 * dr][
                        c + 3 * dc] == color and self.board[r + 4 * dr][
                            c + 4 * dc] == 0:
            # 如果是自己下，且再前面或者再后面一格是空的话，可以找到并返回胜利点
            if color == self.current_player and self.within_boundary(
                    r - dr, c - dc) and self.board[r - dr][c - dc] == 0:
                return True, [r * self.size + c]
            if color == self.current_player and self.within_boundary(
                    r + 5 * dr,
                    c + 5 * dc) and self.board[r + 5 * dr][c + 5 * dc] == 0:
                return True, [(r + 4 * dr) * self.size + c + 4 * dc]
        #如果前后都不空，有两种情况，但是他们都返回相同结果：1.堵对方，那么两边都可以堵 2.自己下但是r-1和r+5都不空，那么两边都可以下
            return True, [
                r * self.size + c, (r + 4 * dr) * self.size + c + 4 * dc
            ]
        return False, []

    def within_boundary(self, r, c):
        return r >= 0 and c >= 0 and r < self.size and c < self.size

    def find_4_match(self, r, c, color, direction):
        count, space_count = 0, 0
        position = []
        for factor in range(5):
            new_r = r + direction[0] * factor
            new_c = c + direction[1] * factor
            cell_value = self.board[new_r][new_c]
            if cell_value == color:
                count = count + 1
            if cell_value == 0:
                space_count = space_count + 1
                position.append(new_r * self.size + new_c)
        if count == 4 and space_count == 1:
            return True, position
        else:
            return False, []

    def move_to_location(self, move):
        if not (0 <= move < self.num_grids):
            raise ValueError(f"Invalid move index: {move}")
        return [move // self.size, move % self.size]

    def location_to_move(self, location):
        if len(location) != 2:
            raise ValueError(f"Invalid location format: {location}")
        r, c = location
        if not self.within_boundary(r, c):
            raise ValueError(f"Location out of bounds: ({r}, {c})")
        return r * self.size + c

    def current_state(self):
        """Return the board state from the perspective of the current player.
        Shape: (4, size, size)
        Channels:
          0 - current player's pieces
          1 - opponent's pieces
          2 - last move
          3 - whose turn (1 for current player, 0 for opponent)
        """

        square_state = np.zeros((4, self.size, self.size), dtype=np.float16)

        for r in range(self.size):
            for c in range(self.size):
                piece = self.board[r][c]
                if piece == self.current_player:
                    square_state[0][r][c] = 1.0
                elif piece == -self.current_player:
                    square_state[1][r][c] = 1.0

        if self.last_move is not None:
            square_state[2][self.last_move // self.size][self.last_move % self.size] = 1.0

        if self.total_moves % 2 == 0:
            square_state[3][:, :] = 1.0

        return square_state[:, ::-1, :]
        # if self.states:
        #     moves, players = np.array(list(zip(*self.states.items())))
        #     move_curr = moves[players == self.current_player]
        #     move_oppo = moves[players != self.current_player]
        #     square_state[0][move_curr // self.width,
        #                     move_curr % self.height] = 1.0
        #     square_state[1][move_oppo // self.width,
        #                     move_oppo % self.height] = 1.0
        #     # indicate the last move location
        #     square_state[2][self.last_move // self.width,
        #                     self.last_move % self.height] = 1.0
        # if len(self.states) % 2 == 0:
        #     square_state[3][:, :] = 1.0  # indicate the color to play
        # return square_state[:, ::-1, :]
        

    def do_move(self, move):
        location = self.move_to_location(move)
        # self.states[move] = self.current_player
        self.board[location[0]][location[1]] = self.current_player
        # self.availables.remove(move)
        self.total_moves = self.total_moves + 1
        # self.current_player = (
        #     self.players[0] if self.current_player == self.players[1]
        #     else self.players[1]
        # )
        self.current_player = self.current_player * -1
        self.last_move = move

    def game_end(self):
        """Check whether the game is ended or not"""
        n = self.n_in_row
        size = self.size

        # game ran too long and too many moves, consider draw
        if self.total_moves > self.draw_threshold * self.num_grids:
            return True, 0
        
        # not enough moves
        if self.total_moves < n * 2 - 1:
            return False, -1

        # check if win
        for r in range(size):
            for c in range(size):
                player = self.board[r][c]
                if player == 0:
                    continue
                if c <= size - n and all(self.board[r][c + i] == player for i in range(n)):
                    return True, player
                if r <= size - n and all(self.board[r + i][c] == player for i in range(n)):
                    return True, player
                if r <= size - n and c <= size - n and all(self.board[r + i][c + i] == player for i in range(n)):
                    return True, player
                if r <= size - n and c >= n - 1 and all(self.board[r + i][c - i] == player for i in range(n)):
                    return True, player
        # if board full, draw
        if self.total_moves >= size * size:
            return True, 0
    
        return False, -1
    
    def print_board(self):
        size = self.size
        
        for x in range(size):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(size - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(size):
                # loc = i * size + j
                p = self.board[i][j]
                if p == self.start_player:
                    print('X'.center(8), end='')
                elif p == -self.start_player:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')        

# class Game(object):
#     """game server"""

#     def start_play(self, board, player1, player2, is_shown=1):
#         """start a game between two players"""
#         players = {1: player1, -1: player2}
#         while True:
#             player_in_turn = players[board.current_player]
#             move, move_probs = player_in_turn.get_action(copy.deepcopy(board), show=True)
#             board.do_move(move)
#             if is_shown:
#                 board.print_board()
#             end, winner = board.game_end()
#             if end:
#                 if is_shown:
#                     if winner != 0:
#                         print("Game end. Winner is", players[winner])
#                     else:
#                         print("Game end. Tie")
#                 return winner

#     def start_evaluate(self, board, player1, player2, is_shown=1):
#         """evaluates two models"""
#         players = {1: player1, -1: player2}
#         while True:
#             player_in_turn = players[board.current_player]
#             act_probs, value = player_in_turn.policy_value_fn(copy.deepcopy(board))
#             move = self.sample_from_top_k_with_ties(act_probs)
#             board.do_move(move)

#             if board.total_moves > 0.7 * board.num_grids:
#                 board.print_board()
#                 print("70% board used. consider tie.")
#                 return 0
            
#             end, winner = board.game_end()
#             if end:
#                 if is_shown:
#                     board.print_board()
#                 return winner

#     def sample_from_top_k_with_ties(self, input_dict, k=3):
#         """
#         从概率前k（含并列）的元素中按归一化概率随机返回一个key。
    
#         参数:
#             input_dict (dict): 形如 {元素: 概率} 的字典。
#             k (int): 前k名（含并列）
    
#         返回:
#             key: 按归一化概率选中的一个key
    
#         异常:
#             ValueError: 如果选中的前k项概率总和为0，则抛出异常。
#         """
#         if isinstance(input_dict, zip):
#             input_dict = dict(input_dict)
        
#         if not input_dict or k <= 0:
#             raise ValueError("输入字典不能为空，且k必须大于0。")
    
#         # 步骤1：获取前k名的概率阈值
#         sorted_probs = sorted(set(input_dict.values()), reverse=True)
#         if len(sorted_probs) < k:
#             threshold = sorted_probs[-1]
#         else:
#             threshold = sorted_probs[k - 1]
    
#         # 步骤2：获取所有满足条件的key及其原始概率
#         top_items = {key: prob for key, prob in input_dict.items() if prob >= threshold}
    
#         # 步骤3：归一化概率
#         total = sum(top_items.values())
#         if total == 0:
#             raise ValueError("前k名（含并列）的元素总概率为0，无法归一化。")
    
#         probs = [v / total for v in top_items.values()]
#         keys = list(top_items.keys())
    
#         # 步骤4：按归一化概率抽样
#         return np.random.choice(keys, p=probs)
    
#     def start_self_play(self, board, player, is_shown=1, temp=1e-3):
#         """ start a self-play game using a MCTS player, reuse the search tree,
#         and store the self-play data: (state, mcts_probs, z) for training
#         """
#         # p1, p2 = board.players
#         states, mcts_probs, current_players = [], [], []
#         while True:
#             move, move_probs = player.get_action(board, temp=temp)
#             # store the data
#             states.append(board.current_state().astype(np.float32))
#             mcts_probs.append(move_probs.astype(np.float32))
#             current_players.append(board.current_player)
#             # perform a move
#             board.do_move(move)
#             # if is_shown:
#             #     self.graphic(board, p1, p2)
#             end, winner = board.game_end()
#             if end:
#                 # winner from the perspective of the current player of each state
#                 winners_z = np.zeros(len(current_players))
#                 if winner != 0:
#                     winners_z[np.array(current_players) == winner] = 1.0
#                     winners_z[np.array(current_players) != winner] = -1.0
#                 # reset MCTS root node
#                 # player.reset_player()
#                 if is_shown:
#                     board.print_board()
#                     # self.graphic(board, p1, p2)
#                     if winner != 0:
#                         print("Game end. Winner is player:", winner)
#                     else:
#                         print("Game end. Tie")
#                 player.reset_player()
#                 return winner, zip(states, mcts_probs, winners_z)
