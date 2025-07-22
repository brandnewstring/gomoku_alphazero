import copy

import numpy as np

c_puct = 5
n_playout = 1000  # number of simulation for each state during mcts
gamma = 0.99 # in gomoku the best strategy should finish the game ASAP. This param devalues the game that wins using too many moves.

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
        self._n_visits = np.uint16(0)
        self._Q = np.float16(0)
        self._u = np.float16(0)
        self._P = np.float16(prior_p)

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits = np.uint16(self._n_visits + 1)
        # Update Q, a running average of values for all visits.
        self._Q = np.float16(self._Q + 1.0 *
                             (leaf_value - self._Q) / self._n_visits)

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = np.float16(
            (c_puct * self._P * np.sqrt(self._parent._n_visits) /
             (1 + self._n_visits)))
        return np.float16(self._Q + self._u)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn

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
            action, node = node.select(c_puct)
            state.do_move(action)
            step_count += 1

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == 0:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.current_player else -1.0)

        # Update value and visit count of nodes in this traversal.
        leaf_value *= gamma**step_count
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSPlayer(object):
    """MCTS Player for self play"""

    def __init__(self, policy_value_function):
        self.mcts = MCTS(policy_value_function)
        # self.policy_value_fn = policy_value_function

    def reset_player(self):
        self._root = TreeNode(None, 1.0)

    # def clear(self):
    #     self.mcts = None

    def get_action(self, board, temp=1e-3, show=False):
        valid_moves = board.get_valid_moves()
        if len(valid_moves) == 0:
            print("WARNING: the board is full")
            return

        # 获取 MCTS 原始概率
        acts, probs = self.mcts.get_move_probs(board, temp)

        # 提取合法动作及概率
        legal_acts_probs = [(a, p) for a, p in zip(acts, probs) if a in valid_moves]
    
        if not legal_acts_probs:
            # fallback 均匀分布
            legal_acts = list(valid_moves)
            legal_probs = [1.0 / len(valid_moves)] * len(valid_moves)
        else:
            legal_acts, legal_probs = zip(*legal_acts_probs)
            legal_acts = list(legal_acts)
            legal_probs = list(legal_probs)
            total_prob = sum(legal_probs)
            if total_prob > 0:
                legal_probs = [p / total_prob for p in legal_probs]
            else:
                legal_probs = [1.0 / len(legal_acts)] * len(legal_acts)
    
        # 构建 move_probs 全局向量
        move_probs = np.zeros(board.size * board.size)
        move_probs[legal_acts] = legal_probs

        # 自对弈：加入 Dirichlet 噪声
        noise = np.random.dirichlet(0.3 * np.ones(len(valid_moves)))
        mixed_probs = 0.85 * move_probs[list(valid_moves)] + 0.15 * noise
        move = np.random.choice(list(valid_moves), p=mixed_probs / mixed_probs.sum())
        self.mcts.update_with_move(move)

        return move, move_probs
