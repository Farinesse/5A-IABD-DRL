import random
from typing import List, Tuple

class TicTacToe:
    def __init__(self):
        self.all_position = [' ' for _ in range(9)]  # Plateau de jeu (9 cases)
        self.terminal_position = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # Lignes
                                  (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Colonnes
                                  (0, 4, 8), (2, 4, 6)]             # Diagonales
        self.current_winner = None  # Suivi du gagnant (X ou O)
        self.agent_position = random.randint(0, 8)  # Position initiale de l'agent

    def reset(self) -> List[str]:
        """Réinitialise le plateau de jeu."""
        self.all_position = [' ' for _ in range(9)]
        self.current_winner = None
        return self.all_position

    def step(self, action: int, letter: str) -> Tuple[List[str], float, bool]:
        """Effectue une action sur le plateau et retourne l'état, la récompense et si la partie est terminée."""
        if self.all_position[action] == ' ':
            self.all_position[action] = letter
            if self.winner(action, letter):
                self.current_winner = letter
            reward = self.score(letter)
            done = self.is_game_over()
            return self.all_position, reward, done
        return self.all_position, 0.0, False

    def available_actions(self) -> List[int]:
        """Renvoie une liste des actions disponibles."""
        return [i for i, spot in enumerate(self.all_position) if spot == ' ']

    def all_states(self) -> List[str]:
        """Renvoie tous les états du plateau de jeu."""
        return self.all_position

    def set_state(self, state: List[str]):
        """Modifie l'état actuel du plateau."""
        self.all_position = state

    def display(self):
        """Affiche le plateau de jeu."""
        for row in [self.all_position[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    def state_id(self) -> List[str]:
        """Renvoie l'état actuel du plateau de jeu."""
        return self.all_position

    def score(self, letter: str) -> float:
        """Calcule la récompense en fonction de la lettre du joueur."""
        if self.current_winner == letter:
            return 1.0  # Récompense positive si le joueur a gagné
        elif self.is_game_over() and not self.current_winner:
            return 0.5  # Récompense neutre en cas de match nul
        else:
            return 0.0  # Récompense nulle si la partie n'est pas terminée

    def is_game_over(self) -> bool:
        """Renvoie True si le jeu est terminé."""
        return self.current_winner is not None or ' ' not in self.all_position

    def winner(self, square: int, letter: str) -> bool:
        """Vérifie si un joueur a gagné."""
        row_ind = square // 3
        row = self.all_position[row_ind * 3:(row_ind + 1) * 3]
        if all([s == letter for s in row]):
            return True
        col_ind = square % 3
        column = [self.all_position[col_ind + i * 3] for i in range(3)]
        if all([s == letter for s in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.all_position[i] for i in [0, 4, 8]]
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.all_position[i] for i in [2, 4, 6]]
            if all([s == letter for s in diagonal2]):
                return True
        return False

    def terminal_states(self) -> List[Tuple[int]]:
        """Renvoie les combinaisons gagnantes (états terminaux)."""
        return self.terminal_position

    def is_forbidden(self, state_or_action: int) -> bool:
        """Retourne False car il n'y a pas d'actions interdites."""
        return False

    def transition_probability(self, state: int, action: int, next_state: int, reward: int) -> float:
        """Retourne toujours 0, comme dans l'exemple LineWorld."""
        return 0.0
