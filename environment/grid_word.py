import random
from typing import List, Tuple

NUM_ACTIONS = 4  # 0: haut, 1: bas, 2: gauche, 3: droite

class GridWorld:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.all_position = [(x, y) for x in range(width) for y in range(height)]
        self.terminal_position = [(0, 0), (width - 1, height - 1)]  # états terminaux aux coins
        self.agent_position = random.choice(self.all_position)  # Position aléatoire de départ
        self.game_over = False  # Etat du jeu
        self.score_val = 0.0  # Score actuel

    def state_description(self) -> List[float]:
        """Renvoie la description de l'état actuel sous forme d'un tableau binaire pour chaque position."""
        return [1.0 if self.agent_position == (x, y) else 0.0 for x in range(self.width) for y in range(self.height)]

    def available_actions_ids(self) -> List[int]:
        """Renvoie une liste des actions disponibles."""
        actions = []
        x, y = self.agent_position
        if y > 0:
            actions.append(0)  # haut
        if y < self.height - 1:
            actions.append(1)  # bas
        if x > 0:
            actions.append(2)  # gauche
        if x < self.width - 1:
            actions.append(3)  # droite
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

        x, y = self.agent_position
        if action == 0 and y > 0:  # haut
            self.agent_position = (x, y - 1)
        elif action == 1 and y < self.height - 1:  # bas
            self.agent_position = (x, y + 1)
        elif action == 2 and x > 0:  # gauche
            self.agent_position = (x - 1, y)
        elif action == 3 and x < self.width - 1:  # droite
            self.agent_position = (x + 1, y)

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
            return -1.0  # Récompense négative pour l'état terminal (0, 0)
        elif self.agent_position == self.terminal_position[1]:
            return 1.0  # Récompense positive pour l'état terminal (width-1, height-1)
        else:
            return 0.0  # Aucune récompense pour les autres états

    def reset(self):
        """Réinitialise la position de l'agent et l'état du jeu."""
        self.agent_position = random.choice(self.all_position)
        self.game_over = False
        self.score_val = 0.0

    def display(self):
        """Affiche le plateau de jeu."""
        for y in range(self.height):
            for x in range(self.width):
                if self.agent_position == (x, y):
                    print('X', end=' ')
                else:
                    print('_', end=' ')
            print()
        print(f"Score : {self.score_val}")
        print(f"Joueur à jouer : {self.agent_position}")
        print(f"Jeu terminé : {self.game_over}")

    def state_id(self) -> Tuple[int, int]:
        return self.agent_position