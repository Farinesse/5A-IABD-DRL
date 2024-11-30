import random
import numpy as np
from typing import List, Tuple
import keras

from algos.DQN.ddqn import double_dqn_no_replay
from algos.DQN.ddqn_exp_replay import double_dqn_with_replay
from algos.DQN.deep_qlearning import deep_q_learning
from algos.PolicyGradientMethods.reinforce_meanbase import REINFORCEBaseline
from functions.outils import plot_csv_data

NUM_ACTIONS = 9
NUM_STATE_FEATURES = 27


class TicTacToe:
    def __init__(self):
        self._board = np.zeros((NUM_ACTIONS,))
        self._player = 0
        self._is_game_over = False
        self._score = 0.0

    def reset(self):
        self._board = np.zeros((NUM_ACTIONS,))
        self._player = 0
        self._is_game_over = False
        self._score = 0.0

    def state_description(self) -> np.ndarray:
        """
        Encode l'état du plateau de TicTacToe dans un vecteur de 27 éléments (3 valeurs par case).
        - [1, 0, 0] pour "X" (Joueur 1).
        - [0, 1, 0] pour "O" (Joueur 2).
        - [0, 0, 1] pour une case vide.
        """
        state_description = np.zeros((NUM_STATE_FEATURES,))

        for i in range(NUM_ACTIONS):
            if self._board[i] == 1.0:  # Case occupée par "X" (Joueur 1)
                state_description[i * 3] = 1.0  # Première des trois valeurs pour "X"
            elif self._board[i] == 2.0:  # Case occupée par "O" (Joueur 2)
                state_description[i * 3 + 1] = 1.0  # Deuxième des trois valeurs pour "O"
            else:
                state_description[i * 3 + 2] = 1.0  # Troisième des trois valeurs pour une case vide
        return state_description

    def available_actions_ids(self) -> np.ndarray:
        return np.where(self._board == 0)[0]

    def action_mask(self) -> np.ndarray:
        return np.where(self._board == 0, 1, 0).astype(np.float32)

    def copy(self):
        """Créer une copie de l'environnement."""
        new_env = TicTacToe()
        new_env._board = self._board.copy()  # Copie du plateau
        new_env._player = self._player  # Copie du joueur courant
        new_env._is_game_over = self._is_game_over  # Copie de l'état de la partie
        return new_env
    def step(self, action: int):
        if self._is_game_over:
            raise ValueError("Game is over, please reset the environment.")

        if action < 0 or action >= NUM_ACTIONS:
            raise ValueError("Invalid move, action must be in [0, 8].")

        if self._board[action] != 0:
            raise ValueError("Invalid move, cell is already occupied.")

        # Mise à jour du plateau
        self._board[action] = self._player + 1

        row = action // 3
        col = action % 3

        # Vérification des lignes
        if self._board[row * 3] == self._board[row * 3 + 1] == self._board[row * 3 + 2]:
            self._is_game_over = True
            self._score = 1.0 if self._player == 0 else -1.0
            return

        # Vérification des colonnes
        if self._board[col] == self._board[col + 3] == self._board[col + 6]:
            self._is_game_over = True
            self._score = 1.0 if self._player == 0 else -1.0
            return

        # Vérification des diagonales si l'action est dans une case de diagonale
        if action in [0, 4, 8]:  # Diagonale principale
            if self._board[0] == self._board[4] == self._board[8]:
                self._is_game_over = True
                self._score = 1.0 if self._player == 0 else -1.0
                return

        if action in [2, 4, 6]:  # Diagonale secondaire
            if self._board[2] == self._board[4] == self._board[6]:
                self._is_game_over = True
                self._score = 1.0 if self._player == 0 else -1.0
                return

        # Vérification du match nul
        if np.all(self._board != 0):
            self._is_game_over = True
            self._score = 0.0
            return

        # Changement de joueur
        self._player = 1 if self._player == 0 else 0

        # Si c'est au tour de l'adversaire aléatoire, il joue automatiquement
        if self._player == 1:
            random_action = np.random.choice(self.available_actions_ids())
            self.step(random_action)

    def is_game_over(self) -> bool:
        return self._is_game_over

    def score(self) -> float:
        return self._score


    def display(self):
        """Affiche le plateau de jeu."""
        for i in range(3):
            print("|".join(["_" if self._board[i * 3 + j] == 0.0 else "X" if self._board[i * 3 + j] == 1.0 else "O" for j in range(3)]))
        print(f"Score : {self._score}")
        print(f"Joueur {'X' if self._player == 0 else 'O'} à jouer")
        print(f"Jeu terminé : {self._is_game_over}")

def create_ttt_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_dim=27),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(9)
    ])
    return model


if __name__ == "__main__":

    env = TicTacToe()
    model = create_ttt_model()
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    """
    trained_model = deep_q_learning(
        model=model,
        target_model=target_model,
        env=env,
        num_episodes=10000,
        gamma=0.99,
        alpha=0.0001,
        start_epsilon=1.0,
        end_epsilon=0.01,
        memory_size=16,
        batch_size=8,
        update_target_steps=100,
        save_path="dqn_model_ttt_test_10000_0-99_0-0001_1-0_0-01_16_8_100.h5",
        input_dim=27
    )

    trained_model, target_model = double_dqn_no_replay(
        online_model=model,
        target_model=target_model,
        env=env,
        num_episodes=10000,
        gamma=0.99,
        alpha=0.0001,
        start_epsilon=1.0,
        end_epsilon=0.01,
        update_target_steps=100,
        save_path="ddqn_model_ttt_test_10000_0-99_0-0001_1-0_0-01_16_8_100.h5",
        input_dim=27
    )

    trained_model, target_model = double_dqn_with_replay(
        online_model=model,
        target_model=target_model,
        env=env,
        num_episodes=10000,
        gamma=0.99,
        alpha=0.0001,
        start_epsilon=1.0,
        end_epsilon=0.01,
        memory_size=16,
        batch_size=8,
        update_target_steps=100,
        save_path="../models/models/ddqn_replay/ddqn_replay_model_ttt_tests/ddqn_with_replay_model_ttt_test_10000_0-99_0-0001_1-0_0-01_16_8_100.h5",
        input_dim=27
    )


    plot_csv_data(
        "../models/models/ttt/ddqn_model_ttt_test_10000_0-99_0-0001_1-0_0-01_16_8_100.h5_metrics.csv")
    """

    """trained_model = deep_q_learning(
        model=model,
        target_model=target_model,
        env=env,
        num_episodes=10000,
        gamma=0.99,
        alpha=0.0001,
        start_epsilon=1.0,
        end_epsilon=0.01,
        memory_size=32,
        batch_size=16,
        update_target_steps=100,
        save_path ='dqn_replay_model_ttt_test_10000_0-99_0-0001_1-0_32_16_0-01_16_8_100.h5',
        input_dim=27,
    )"""

    agent = REINFORCEBaseline(
        state_dim=27,
        action_dim=9,
        alpha_theta=0.0001,
        alpha_w=0.001,
        gamma=0.99,
        path='ReinforceBaseline_model_ttt_test_10000_0-99_0-0001_0-001_128_512_256.h5_metrics'
    )

    agent.train(env, episodes=10000)
    plot_csv_data(agent.path + "_metrics.csv")

    #plot_csv_data("dqn_replay_model_ttt_test_10000_0-99_0-0001_1-0_32_16_0-01_16_8_100.h5_metrics.csv")