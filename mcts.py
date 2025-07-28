import copy
import math
import time

import numpy as np

from config import board_size, draw_threshold

c_puct = 5 # a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
n_playout = 1000  # number of simulation for each state during mcts
gamma = 0.98 # in gomoku the best strategy should finish the game ASAP. This param devalues the game that wins using too many moves.
max_depth = math.ceil(board_size*board_size*draw_threshold) + 1
gamma_powers = [gamma**i for i in range(max_depth + 1)]
epsilon = 0.25
alpha = 10 / (board_size * board_size)
temp = 1.0  # the temperature param


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    __slots__ = ("_parent", "_children", "_n_visits", "_Q", "_u", "_P")

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = prior_p

    def expand(self, priors, legal_positions):
        """Expand tree by creating new children. It also adds dirichlet noise as AlphaZero.
         priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # 根节点加 Dirichlet 噪声
        if self.is_root():
            noise = np.random.dirichlet([alpha] * len(legal_positions))
            priors = (1 - epsilon) * priors + epsilon * noise
    
        # 建立子节点
        for a, p in zip(legal_positions, priors):
            if a not in self._children:
                self._children[a] = TreeNode(self, p)
            else:
                raise ValueError(f"Child node for action {a} already exists!")

    def select(self):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value())

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits = self._n_visits + 1
        # Update Q, a running average of values for all visits.
        self._Q = self._Q + 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        """
        self._u = (c_puct * self._P * math.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return not self._children

    def is_root(self):
        return self._parent is None


class MCTSPlayer(object):
    """A Monte Carlo Tree Search Player who holds a MCTS tree."""

    def __init__(self, policy_value_fn):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        """
        self._root = TreeNode(None, 1.0)
        self._policy_value_fn = policy_value_fn

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        step_count = 0
        while (1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select()
            state.do_move(action)
            step_count += 1

        end, winner = state.game_end()
        if end:
            # if game already ends, no need to predict using network
            if winner == 0:
                return
            else:
                leaf_value = 1.0 if winner == state.current_player else -1.0
                leaf_value *= gamma_powers[step_count]
        else:
            # if game not end, we need to expand. we also need to use the predicted value
            # to update the parents because it's the 1st time expand the node.
            legal_probs, leaf_value, legal_positions = self._policy_value_fn(state)
            node.expand(legal_probs, legal_positions)

        node.update_recursive(-leaf_value)

    def get_action(self, state, show=False):
        """
        Gets the next move using mcts given a board state. Returns the move and the move probabilities for the whole board.
        """
        valid_moves = np.array(state.get_valid_moves(), dtype=np.int32)

        # --- 1. MCTS playout ---
        n = n_playout
        while n > 0:
            n -= 1
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
    
        # --- 2. 直接构造合法落子的访问次数 ---
        visits = np.array(
            [self._root._children.get(a)._n_visits if a in self._root._children else 0 for a in valid_moves],
            dtype=np.float32
        )
    
        # --- 3. 转成概率（softmax） ---
        if visits.sum() > 0:
            scaled = np.log(visits + 1e-10) / temp
            legal_probs = softmax(scaled)
        else:
            legal_probs = np.ones(len(valid_moves), dtype=np.float32) / len(valid_moves)
    
        # --- 4. 采样动作（仅在合法落子集合中） ---
        move = np.random.choice(valid_moves, p=legal_probs)
    
        # --- 5. 构造完整棋盘概率向量 ---
        full_probs = np.zeros(board_size * board_size, dtype=np.float32)
        full_probs[valid_moves] = legal_probs
    
        # --- 6. 更新树 ---
        self.update_with_move(move)
    
        return move, full_probs
    
    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        self._root = self._root._children[last_move]
        self._root._parent = None

    def reset_player(self):
        self._root = TreeNode(None, 1.0)
