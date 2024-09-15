import random
from typing import List, Tuple


class LineWorld:
    def __init__(self, length: int):
        self.all_position = list(range(length))
        self.terminal_position = [0, length - 1]
        self.all_actions = [0, 1, 2]  # 0: stay, 1: left, 2: right
        self.agent_position = random.randint(0, length - 1)

    def random_state(self):
        # Pas d'implémentation demandée dans Rust
        pass

    def reset(self) -> int:
        self.agent_position = random.randint(0, len(self.all_position) - 1)
        return self.agent_position

    def step(self, action: int) -> Tuple[int, float, bool]:
        if action == 1 and self.agent_position > 0:
            self.agent_position -= 1
        elif action == 2 and self.agent_position < len(self.all_position) - 1:
            self.agent_position += 1

        reward = self.score()
        done = self.is_game_over()
        return self.agent_position, reward, done

    def available_actions(self) -> List[int]:
        actions = [0]  # L'action de rester sur place
        if self.agent_position > 0:
            actions.append(1)  # Se déplacer à gauche
        if self.agent_position < len(self.all_position) - 1:
            actions.append(2)  # Se déplacer à droite
        return actions

    def all_states(self) -> List[int]:
        return self.all_position

    def set_state(self, state: int):
        self.agent_position = state

    def display(self):
        game = ''.join('X' if pos == self.agent_position else '_' for pos in self.all_position)
        print(game)

    def state_id(self) -> int:
        return self.agent_position

    def score(self) -> float:
        if self.agent_position == self.terminal_position[0]:
            return -1.0  # Récompense négative pour l'état terminal à gauche
        elif self.agent_position == self.terminal_position[1]:
            return 1.0  # Récompense positive pour l'état terminal à droite
        else:
            return 0.0  # Aucune récompense pour les autres états

    def is_game_over(self) -> bool:
        return self.agent_position in self.terminal_position

    def all_action(self) -> List[int]:
        return self.all_actions

    def terminal_states(self) -> List[int]:
        return self.terminal_position

    def is_forbidden(self, state_or_action: int) -> bool:
        # Toujours retourner False comme dans l'exemple Rust
        return False

    def transition_probability(self, state: int, action: int, next_state: int, reward: int) -> float:
        # Retourne toujours 0 comme dans l'exemple Rust
        return 0.0



