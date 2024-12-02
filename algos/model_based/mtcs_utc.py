import os
import pickle

import numpy as np
from math import sqrt, log
from typing import Dict, Any

import pandas as pd
from tqdm import tqdm

from environment.FarkelEnv import FarkleDQNEnv
from environment.grid_word import GridWorld
from environment.line_word import LineWorld
from environment.tictactoe import TicTacToe
from functions.outils import log_metrics_to_dataframe_mcts, play_with_mcts, plot_mcts_metrics


class Node:
    def __init__(self, state: Any, parent=None, action=None):
        self.state = state.copy() if hasattr(state, 'copy') else state
        self.parent = parent
        self.action = action
        self.children: Dict[Any, Node] = {}
        self.visits = 0
        self.value = 0.0
        self.player = 0 if parent is None else 1 - parent.player

    def get_ucb(self, action: Any, C: float = sqrt(2)) -> float:
        """Calcule le score UCB pour une action."""
        if action not in self.children or self.children[action].visits == 0:
            return float('inf')

        child = self.children[action]
        exploitation = child.value / child.visits
        exploration = C * sqrt(log(self.visits) / child.visits)

        return exploitation + exploration


class MCTS:
    def __init__(self, n_simulations: int = 2000, C: float = sqrt(2)):
        self.root = None
        self.n_simulations = n_simulations
        self.C = C
        self.actions_taken = []  # Liste pour sauvegarder les actions prises

    def select_action(self, env) -> Any:
        """Sélectionne la meilleure action selon MCTS."""
        root = Node(env.state_description())

        for _ in range(self.n_simulations):
            node = root
            sim_env = self._copy_env(env)

            # Selection
            while not sim_env.is_game_over() and self._is_fully_expanded(node, sim_env):
                action = self._select_ucb_action(node, sim_env)
                sim_env.step(action)
                node = node.children[action]

            # Expansion
            if not sim_env.is_game_over():
                action = self._get_unexplored_action(node, sim_env)
                sim_env.step(action)
                node = self._add_child(node, sim_env, action)

            # Simulation
            value = self._rollout(sim_env)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent

        # Sélection de la meilleure action
        valid_actions = env.available_actions_ids()
        if len(valid_actions) == 0:
            return -1

        # Choisir la meilleure action
        best_action = max(valid_actions,
                          key=lambda a: root.children[a].visits if a in root.children else float('-inf'))

        # Sauvegarder l'action choisie
        self.actions_taken.append(best_action)
        return best_action

    def _copy_env(self, env) -> Any:
        """Crée une copie de l'environnement."""
        if isinstance(env, LineWorld):
            new_env = LineWorld(len(env.all_position))
            new_env.agent_position = env.agent_position
            new_env.all_position = env.all_position.copy()
            new_env.terminal_position = env.terminal_position.copy()
            new_env.game_over = env.game_over
            new_env.score_val = env.score_val
        elif isinstance(env, TicTacToe):
            new_env = TicTacToe()
            new_env._board = env._board.copy()
            new_env._player = env._player
            new_env._is_game_over = env._is_game_over
        elif isinstance(env,FarkleDQNEnv) :
            new_env = FarkleDQNEnv(env.num_players, env.target_score)
            new_env.scores = env.scores.copy()
            new_env.current_player = env.current_player
            new_env.round_score = env.round_score
            new_env.remaining_dice = env.remaining_dice
            new_env.dice_roll = env.dice_roll.copy()
            new_env.game_over = env.game_over
            new_env.last_action_stop = env.last_action_stop
        elif isinstance(env,GridWorld) :
            new_env = GridWorld(env.width, env.height)
            new_env.agent_position = env.agent_position
            new_env.all_position = env.all_position.copy()
            new_env.terminal_position = env.terminal_position.copy()
            new_env.game_over = env.game_over
            new_env.score_val = env.score_val
        return new_env

    def _is_fully_expanded(self, node: Node, env) -> bool:
        """Vérifie si tous les coups valides ont été explorés."""
        valid_actions = env.available_actions_ids()
        return all(action in node.children for action in valid_actions)

    def _select_ucb_action(self, node: Node, env) -> Any:
        """Sélectionne une action selon UCB."""
        valid_actions = env.available_actions_ids()
        return max(valid_actions, key=lambda a: node.get_ucb(a, self.C))

    def _get_unexplored_action(self, node: Node, env) -> Any:
        """Retourne une action non explorée aléatoire."""
        valid_actions = env.available_actions_ids()
        unexplored = [a for a in valid_actions if a not in node.children]
        return np.random.choice(unexplored)

    def _add_child(self, parent: Node, env, action: Any) -> Node:
        """Ajoute un nouveau nœud enfant."""
        child = Node(env.state_description(), parent, action)
        parent.children[action] = child
        return child

    def _rollout(self, env) -> float:
        """Effectue une simulation aléatoire jusqu'à état terminal."""
        while not env.is_game_over():
            valid_actions = env.available_actions_ids()
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            env.step(action)
        return env.score()

    def train(self, env, episodes=1000, interval=100, save_path=None):
        """Entraîne l'agent et enregistre les métriques."""
        results_df = None

        for episode in tqdm(range(episodes), desc="Training MCTS"):
            if episode % interval == 0 and episode > 0:
                results_df = log_metrics_to_dataframe_mcts(
                    function=play_with_mcts,
                    agent=self,
                    env=env,
                    episode_index=episode,
                    games=10,
                    dataframe=results_df
                )

                if save_path:
                    # Sauvegarde du modèle
                    self.save(f"{save_path}_ep{episode}.pkl")
                    # Sauvegarde des métriques
                    results_df.to_csv(f"{save_path}_metrics.csv", index=False)

            env.reset()
            while not env.is_game_over():
                action = self.select_action(env)
                if action == -1:
                    break
                env.step(action)

        # Sauvegarde finale
        if save_path:
            self.save(f"{save_path}_final.pkl")
            if results_df is not None:
                results_df.to_csv(f"{save_path}_final_metrics.csv", index=False)

        return results_df


    def save(self, filepath: str):
        """Sauvegarde le modèle MCTS et les actions prises"""
        # Crée le dossier si il n'existe pas
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Création du dictionnaire contenant l'état du modèle
        model_state = {
            'n_simulations': self.n_simulations,
            'C': self.C,
            'root': self.root,
            'actions_taken': self.actions_taken  # Sauvegarde des actions prises
        }

        # Sauvegarde dans le fichier
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

        print(f"Modèle et actions sauvegardés dans {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Charge un modèle MCTS sauvegardé"""
        # 1. Lecture du fichier binaire
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)

        # 2. Création d'une nouvelle instance MCTS
        mcts = cls(
            n_simulations=model_state['n_simulations'],
            C=model_state['C']
        )

        # 3. Restauration de l'arbre et des actions
        mcts.root = model_state['root']
        mcts.actions_taken = model_state['actions_taken']
        return mcts



if __name__ == "__main__":
    # Configuration
    #env = LineWorld(10)
    env = TicTacToe()
    #env = GridWorld(5, 5)
    #env = FarkleDQNEnv(target_score=5000)

    save_path = f"models/mcts_{env.__class__.__name__}_100_sims"


    mcts_agent = MCTS(n_simulations=20, C=1.2)
    metrics_df = mcts_agent.train(
        env=env,
        episodes=100,
        interval=10,
        save_path=save_path
    )

    # Plot des résultats
    plot_mcts_metrics(
        f"{save_path}_final_metrics.csv",
        title=f"MCTS Training on {env.__class__.__name__}"
    )