import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    tf.keras.backend.set_value(optimizer.lr, lr)


class PolicyValueNet:
    """policy-value network"""

    def __init__(self, board_size, model_file=None):
        self.board_size = board_size
        self.l2_const = 1e-4  # L2 regularization coefficient

        l2_reg = regularizers.l2(self.l2_const)

        # ===== Input layer =====
        inputs = layers.Input(shape=(4, board_size, board_size))
        x = layers.Permute((2, 3, 1))(inputs)  # (N, H, W, C)

        # ===== Initial conv =====
        x = layers.Conv2D(128, 3, padding='same', use_bias=False,
                          kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # ===== Residual blocks =====
        def residual_block(x, filters):
            shortcut = x
            x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                              kernel_regularizer=l2_reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                              kernel_regularizer=l2_reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            return x

        # Stack N residual blocks (e.g., 5 blocks)
        for _ in range(5):  # 你也可以改成10、15
            x = residual_block(x, filters=128)

        # ===== Policy head =====
        x_act = layers.Conv2D(2, 1, use_bias=False,
                              kernel_regularizer=l2_reg)(x)
        x_act = layers.BatchNormalization()(x_act)
        x_act = layers.Activation('relu')(x_act)
        x_act = layers.Flatten()(x_act)
        x_act = layers.Dense(board_size * board_size,
                             activation='softmax',
                             kernel_regularizer=l2_reg)(x_act)

        # ===== Value head =====
        x_val = layers.Conv2D(1, 1, use_bias=False,
                              kernel_regularizer=l2_reg)(x)
        x_val = layers.BatchNormalization()(x_val)
        x_val = layers.Activation('relu')(x_val)
        x_val = layers.Flatten()(x_val)
        x_val = layers.Dense(64, activation='relu',
                             kernel_regularizer=l2_reg)(x_val)
        x_val = layers.Dense(1, activation='tanh',
                             kernel_regularizer=l2_reg)(x_val)

        # ===== Model =====
        self.model = models.Model(inputs=inputs, outputs=[x_act, x_val])
        self.optimizer = optimizers.Adam()

        self.model.compile(
            optimizer=self.optimizer,
            loss=['categorical_crossentropy', 'mean_squared_error'],
            loss_weights=[1.0, 0.01]  # 可调
        )

        if model_file:
            self.model.load_weights(model_file)


    def policy_value(self, state_batch):
        """Predict policy (probabilities) and value for a batch of states"""
        act_probs, value = self.model.predict(state_batch, verbose=0)
        return act_probs, value

    def policy_value_fn(self, board):
        """Predict the action probabilities and value for a board state"""
        legal_positions = board.get_valid_moves()
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_size, self.board_size))
        # act_probs, value = self.model.predict(current_state, verbose=0)
        act_probs, value = self.model(current_state, training=False)
        # act_probs = act_probs.flatten()
        act_probs = act_probs.numpy().flatten()
        value = value.numpy()[0][0]
        # value = value[0][0]
        # act_probs = zip(legal_positions, act_probs[legal_positions])
        legal_probs = act_probs[legal_positions]
        
        return legal_probs, value, legal_positions

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """Perform a training step"""
        set_learning_rate(self.optimizer, lr)
        # train on batch
        loss = self.model.train_on_batch(state_batch,
                                         [mcts_probs, winner_batch])
        # entropy (for monitoring)
        pred_probs, _ = self.model.predict(state_batch, verbose=0)
        entropy = -np.mean(
            np.sum(pred_probs * np.log(np.clip(pred_probs, 1e-10, 1.0)),
                   axis=1))
        return loss[0], entropy

    def get_policy_param(self):
        """Get model weights"""
        return self.model.get_weights()

    def save_model(self, model_file):
        """Save model parameters"""
        self.model.save_weights(model_file)
