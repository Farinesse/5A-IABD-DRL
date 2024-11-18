import random
from typing import List

NUM_ACTIONS = 3  # 0: rester sur place, 1: gauche, 2: droite

class LineWorld:
    def __init__(self, length: int):
        self.all_position = list(range(length))
        self.terminal_position = [0, length - 1]
        self.agent_position = random.randint(0, length - 1)
        self.game_over = False  # Indique si le jeu est terminé
        self.score_val = 0.0  # Score actuel

    def state_description(self) -> List[float]:
        """Renvoie la description de l'état actuel sous forme d'un tableau binaire."""
        return [1.0 if pos == self.agent_position else 0.0 for pos in self.all_position]

    def available_actions_ids(self) -> List[int]:
        """Renvoie une liste des actions disponibles."""
        actions = [0]  # L'action de rester sur place
        if self.agent_position > 0:
            actions.append(1)  # Se déplacer à gauche
        if self.agent_position < len(self.all_position) - 1:
            actions.append(2)  # Se déplacer à droite
        return actions

    def action_mask(self) -> List[float]:
        """Renvoie un masque binaire indiquant les actions possibles."""
        mask = [0.0] * NUM_ACTIONS
        for action in self.available_actions_ids():
            mask[action] = 1.0
        return mask

    def step(self, action: int):
        """Effectue une action et met à jour l'état du jeu."""
        if self.game_over:
            raise ValueError("Le jeu est terminé, aucune action ne peut être effectuée.")

        if action == 1 and self.agent_position > 0:
            self.agent_position -= 1
        elif action == 2 and self.agent_position < len(self.all_position) - 1:
            self.agent_position += 1

        # Mise à jour du score
        self.score_val = self.score()

        # Vérification si le jeu est terminé
        self.game_over = self.is_game_over()

    def is_game_over(self) -> bool:
        """Retourne True si la partie est terminée."""
        return self.agent_position in self.terminal_position

    def score(self) -> float:
        """Renvoie la récompense actuelle."""
        if self.agent_position == self.terminal_position[0]:
            return -1.0  # Récompense négative pour l'état terminal à gauche
        elif self.agent_position == self.terminal_position[1]:
            return 1.0  # Récompense positive pour l'état terminal à droite
        else:
            return 0.0  # Aucune récompense pour les autres états

    def reset(self):
        """Réinitialise la position de l'agent et l'état du jeu."""
        self.agent_position = random.randint(0, len(self.all_position) - 1)
        self.game_over = False
        self.score_val = 0.0

    def display(self):
        """Affiche l'état actuel du LineWorld."""
        game = ''.join('X' if pos == self.agent_position else '_' for pos in self.all_position)
        print(game)
        print(f"Score : {self.score_val}")
        print(f"Joueur à jouer : {self.agent_position}")
        print(f"Jeu terminé : {self.game_over}")

    def state_id(self) -> int:
        return self.agent_position

