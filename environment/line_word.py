import random
import numpy as np
from typing import List, Tuple, Any
import keras
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit

from algos.DQN.ddqn import double_dqn_no_replay
from algos.DQN.deep_qlearning import deep_q_learning
from algos.DQN.dqn import dqn_no_replay
from algos.PolicyGradientMethods.one_step_actor_critic import OneStepActorCritic
from algos.PolicyGradientMethods.ppo import PPOActorCritic
from algos.PolicyGradientMethods.reinforce import REINFORCE
from algos.PolicyGradientMethods.reinforceBC import REINFORCEWithCritic
from algos.PolicyGradientMethods.reinforce_meanbase import REINFORCEMeanBaseline
from functions.outils import plot_csv_data

NUM_ACTIONS = 3  # 0: rester sur place, 1: gauche, 2: droite

class LineWorld:
    def __init__(self, length: int):
        self.all_position = list(range(length))
        self.terminal_position = [0, length - 1]
        self.agent_position = random.randint(1, length - 2)
        self.game_over = False  # Indique si le jeu est terminé
        self.score_val = 0.0  # Score actuel

    def env_description(self) -> str:
        """Renvoie une description de l'environnement."""
        return f"LineWorld({len(self.all_position)})"

    def state_description(self) -> np.ndarray:
        """
        Returns the current state as a one-hot vector of length `self.length`.
        """
        state = np.zeros((len(self.all_position)))
        state[self.agent_position] = 1.0
        return state

    def available_actions_ids(self) -> np.ndarray:
        """Renvoie une liste des actions disponibles."""
        actions = [0]  # L'action de rester sur place
        if self.agent_position > 0:
            actions.append(1)  # Se déplacer à gauche
        if self.agent_position < len(self.all_position) - 1:
            actions.append(2)  # Se déplacer à droite
        return np.array(actions)

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

    def score(self, testing=None) -> float:
        """Renvoie la récompense actuelle."""
        if self.agent_position == self.terminal_position[0]:
            return -1.0  # Récompense négative pour l'état terminal à gauche
        elif self.agent_position == self.terminal_position[1]:
            return 1.0  # Récompense positive pour l'état terminal à droite
        else:
            return 0.0  # Aucune récompense pour les autres états

    def reset(self):
        """Réinitialise la position de l'agent et l'état du jeu."""
        self.agent_position = random.randint(1, len(self.all_position) - 2)
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

def create_line_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_dim=10),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(3)
    ])
    return model

if __name__ == "__main__":
    env = LineWorld(10)
    model = create_line_model()
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())


    """trained_model, target_model = double_dqn_no_replay(
        online_model=model,
        target_model=target_model,
        env=env,
        num_episodes=100000,
        gamma=0.99,
        alpha=0.0001,
        start_epsilon=1.0,
        end_epsilon=0.01,
        update_target_steps=1000,
        save_path="ddqn_noreplay_lineworld",
        input_dim=10,
        interval=1000
    )"""


    """trained_model, target_model = dqn_no_replay(
        model=model,
        target_model=target_model,
        env=env,
        num_episodes=100000,
        gamma=0.99,
        start_epsilon=1.0,
        end_epsilon=0.01,
        update_frequency=1000,
        save_path="dqn_noreplay_lineworld",
        input_dim=10,
        interval=1000
    )"""

    """trained_model = deep_q_learning(
        model=model,
        target_model=target_model,
        env=env,
        num_episodes=100000,
        gamma=0.99,
        alpha=0.001,
        start_epsilon=1.0,
        end_epsilon=0.01,
        memory_size=32,
        batch_size=16,
        update_target_steps=1000,
        save_path ='dqn_replay_lineworld',
        input_dim=10,
        interval=1000
    )"""

    """REINFORCEMeanBaseline(
        state_dim=10,
        action_dim=3,
        alpha=0.0001,
        gamma=0.99,
        path='reinforce_mb_line'
    ).train(env, episodes=10000, interval=100)"""

    """REINFORCE(
        state_dim=10,
        action_dim=3,
        alpha=0.0001,
        gamma=0.99,
        path='reinforce_line'
    ).train(env, episodes=10000, interval=100)"""

    """REINFORCEWithCritic(
        state_dim=10,
        action_dim=3,
        alpha_theta=0.0001,
        alpha_w=0.0001,
        gamma=0.99,
        path='reinforce_mb_critic_line'
    ).train(env, episodes=10000, interval=100)"""

    """PPOActorCritic(
        state_dim=10,
        action_dim=3,
        clip_epsilon=0.2,
        gamma=0.99,
        alpha=0.0001,
        path='ppo_line'
    ).train(env, episodes=10000, interval=100)"""

    OneStepActorCritic(
        state_dim=10,
        action_dim=3,
        alpha_theta=0.0001,
        alpha_w=0.0001,
        gamma=0.99,
        path='1_step_actor_critic_line'
    ).train(env, episodes=1000, interval=100)
