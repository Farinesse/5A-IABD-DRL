import random
import numpy as np
from typing import List, Tuple

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

if __name__ == "__main__":
    game = TicTacToe()
    game.display()
    while not game.is_game_over():
        if game._player == 0:  # Tour de l'agent
            print("\nTour de l'agent (joueur X) :")
            action = int(input("Entrez une action (0-8) : "))
            game.step(action)
            game.display()
            print(game.available_actions_ids())
            print(game.state_description())
        else:  # Tour de l'adversaire
            print("\nTour de l'adversaire (joueur O) :")
