import random
from typing import List, Tuple

NUM_ACTIONS = 9

class TicTacToe_new:
    def __init__(self):
        self.board = [0.0 for _ in range(NUM_ACTIONS)]  # Plateau de 9 cases
        self.player = 0  # Le joueur 0 (X) joue en premier
        self.score_val = 0.0  # Renommé en score_val pour éviter le conflit
        self.game_over = False  # Indique si le jeu est terminé
        self.current_winner = None  # Suivi du gagnant

    def state_description(self) -> List[float]:
        """Renvoie la description de l'état actuel du plateau sous forme de tableau de 27 cases (3 états possibles par case)."""
        return [1.0 if self.board[cell] == feature else 0.0 for cell in range(9) for feature in range(3)]

    def available_actions_ids(self) -> List[int]:
        """Renvoie une liste d'actions disponibles (cases vides)."""
        return [idx for idx, val in enumerate(self.board) if val == 0.0]

    def action_mask(self) -> List[float]:
        """Renvoie un masque binaire indiquant les actions possibles."""
        return [1.0 if val == 0.0 else 0.0 for val in self.board]

    def step(self, action: int):
        """Effectue une action et met à jour l'état du jeu."""
        if self.game_over:
            raise ValueError("Le jeu est terminé, aucune action ne peut être effectuée.")

        if action >= NUM_ACTIONS or self.board[action] != 0.0:
            raise ValueError(f"Action invalide : {action}")

        # Le joueur courant effectue l'action
        self.board[action] = float(self.player + 1)

        # Vérification des lignes, colonnes, et diagonales
        row = action // 3
        col = action % 3

        if (self.board[row * 3] == self.board[row * 3 + 1] == self.board[row * 3 + 2] or
            self.board[col] == self.board[col + 3] == self.board[col + 6] or
            self.board[0] == self.board[4] == self.board[8] == self.board[action] or
            self.board[2] == self.board[4] == self.board[6] == self.board[action]):
            self.game_over = True
            self.current_winner = self.player
            self.score_val = 1.0 if self.player == 0 else -1.0
            return

        # Si toutes les cases sont remplies, match nul
        if all(val != 0.0 for val in self.board):
            self.game_over = True
            self.score_val = 0.0  # Match nul
            return

        # Changement de joueur
        self.player = 1 if self.player == 0 else 0

    def is_game_over(self) -> bool:
        """Retourne True si la partie est terminée."""
        return self.game_over

    def score(self) -> float:
        """Renvoie le score actuel."""
        return self.score_val

    def reset(self):
        """Réinitialise le plateau pour une nouvelle partie."""
        self.board = [0.0 for _ in range(NUM_ACTIONS)]
        self.player = 0
        self.score_val = 0.0
        self.game_over = False
        self.current_winner = None

    def state_id(self) -> Tuple[float]:
        """Renvoie un identifiant unique pour l'état du plateau sous forme de tuple immuable."""
        return tuple(self.board)

    def display(self):
        """Affiche le plateau de jeu."""
        for i in range(3):
            print("|".join(["_" if self.board[i * 3 + j] == 0.0 else "X" if self.board[i * 3 + j] == 1.0 else "O" for j in range(3)]))
        print(f"Score : {self.score_val}")
        print(f"Joueur {self.player} à jouer")
        print(f"Jeu terminé : {self.game_over}")
