import random
from typing import List, Tuple


class GridWorld:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.all_position = [(x, y) for x in range(width) for y in range(height)]
        self.terminal_position = [(0, 0), (width - 1, height - 1)]  # états terminaux aux coins
        self.agent_position = random.choice(self.all_position)  # Position aléatoire de départ
        self.all_actions = [0, 1, 2, 3]  # 0: haut, 1: bas, 2: gauche, 3: droite

    def reset(self) -> Tuple[int, int]:
        self.agent_position = random.choice(self.all_position)
        return self.agent_position

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        x, y = self.agent_position

        if action == 0 and y > 0:  # haut
            self.agent_position = (x, y - 1)
        elif action == 1 and y < self.height - 1:  # bas
            self.agent_position = (x, y + 1)
        elif action == 2 and x > 0:  # gauche
            self.agent_position = (x - 1, y)
        elif action == 3 and x < self.width - 1:  # droite
            self.agent_position = (x + 1, y)

        reward = self.score()
        done = self.is_game_over()
        return self.agent_position, reward, done

    def available_actions(self) -> List[int]:
        x, y = self.agent_position
        actions = []
        if y > 0:
            actions.append(0)  # haut
        if y < self.height - 1:
            actions.append(1)  # bas
        if x > 0:
            actions.append(2)  # gauche
        if x < self.width - 1:
            actions.append(3)  # droite
        return actions

    def display(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.agent_position == (x, y):
                    print('X', end=' ')
                else:
                    print('_', end=' ')
            print()  # Nouvelle ligne après chaque rangée

    def state_id(self) -> Tuple[int, int]:
        return self.agent_position

    def score(self) -> float:
        if self.agent_position == self.terminal_position[0]:
            return -1.0  # Récompense négative pour l'état terminal (0, 0)
        elif self.agent_position == self.terminal_position[1]:
            return 1.0  # Récompense positive pour l'état terminal (width-1, height-1)
        else:
            return 0.0  # Aucune récompense pour les autres états

    def is_game_over(self) -> bool:
        return self.agent_position in self.terminal_position

    def terminal_states(self) -> List[Tuple[int, int]]:
        return self.terminal_position
