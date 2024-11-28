import numpy as np
from math import sqrt, log
from typing import Dict
from tqdm import tqdm
from environment.tictactoe import TicTacToe


class Node:
    def __init__(self, state: np.ndarray, parent=None, prior_action=None):
        self.state = state.copy()
        self.parent = parent
        self.prior_action = prior_action
        self.children: Dict[int, Node] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.player = 0 if parent is None else 1 - parent.player

    def is_fully_expanded(self, valid_actions):
        """Vérifie si tous les coups valides ont été explorés."""
        return all(action in self.children for action in valid_actions)

    def get_ucb_score(self, action: int, C: float = sqrt(2)) -> float:
        """Calcule le score UCB pour une action."""
        if action not in self.children or self.children[action].visit_count == 0:
            return float('inf')

        exploitation = self.children[action].total_value / self.children[action].visit_count
        exploration = C * sqrt(log(self.visit_count) / self.children[action].visit_count)

        return exploitation + exploration


class MCTS:
    def __init__(self, n_simulations: int = 1000, C: float = sqrt(2)):
        self.n_simulations = n_simulations
        self.C = C

    def _copy_env(self, env):
        """Crée une copie de l'environnement."""
        new_env = TicTacToe()
        new_env._board = env._board.copy()
        new_env._player = env._player
        new_env._is_game_over = env._is_game_over
        new_env._score = env._score
        return new_env

    def select_action(self, env) -> int:
        """Sélectionne la meilleure action après avoir effectué les simulations MCTS."""
        root = Node(env._board)
        root.player = env._player

        for _ in range(self.n_simulations):
            node = root
            sim_env = self._copy_env(env)

            # Sélection
            while not sim_env.is_game_over() and node.is_fully_expanded(sim_env.available_actions_ids()):
                valid_actions = sim_env.available_actions_ids()
                if not valid_actions.size:
                    break

                action = max(valid_actions, key=lambda a: node.get_ucb_score(a, self.C))
                sim_env.step(action)
                node = node.children[action]

            # Expansion
            if not sim_env.is_game_over():
                valid_actions = sim_env.available_actions_ids()
                for action in valid_actions:
                    if action not in node.children:
                        sim_env_copy = self._copy_env(sim_env)
                        sim_env_copy.step(action)
                        node.children[action] = Node(sim_env_copy._board, node, action)
                        node = node.children[action]
                        break

            # Simulation
            sim_env_rollout = self._copy_env(sim_env)
            while not sim_env_rollout.is_game_over():
                valid_actions = sim_env_rollout.available_actions_ids()
                if not valid_actions.size:
                    break
                action = np.random.choice(valid_actions)
                sim_env_rollout.step(action)

            # Backpropagation
            value = sim_env_rollout.score()
            while node is not None:
                node.visit_count += 1
                node.total_value += value if node.player == 0 else -value
                node = node.parent

        # Sélection de la meilleure action
        valid_actions = env.available_actions_ids()
        if not valid_actions.size:
            return -1

        # Choisir l'action la plus visitée
        best_action = max(valid_actions,
                          key=lambda a: root.children[a].visit_count if a in root.children else -float('inf'))
        return best_action

    # Dans la classe MCTS, modifier la méthode train :
    def train(self, env: TicTacToe, episodes: int = 1000):
        """Évalue les performances de l'agent sur plusieurs épisodes."""
        history = []
        win_count = 0
        draw_count = 0
        total_score = 0

        for episode in tqdm(range(episodes), desc="Training MCTS"):
            env.reset()
            game_over = False

            while not game_over:
                if env._player == 0:  # Tour du MCTS
                    action = self.select_action(env)
                    if action == -1:
                        break
                    env.step(action)
                game_over = env.is_game_over()

            # Enregistre le résultat
            score = env.score()
            history.append(score)
            total_score += score

            if score > 0:
                win_count += 1
            elif score == 0:
                draw_count += 1

            # Affiche les statistiques tous les 100 épisodes
            if (episode + 1) % 1000 == 0:
                win_rate = win_count / (episode + 1)
                draw_rate = draw_count / (episode + 1)
                avg_score = total_score / (episode + 1)
                print(f"\nÉpisode {episode + 1}")
                print(f"Taux de victoire: {win_rate:.2%}")
                print(f"Taux de match nul: {draw_rate:.2%}")
                print(f"Score moyen global: {avg_score:.3f}")
                print(f"Score moyen derniers 100: {np.mean(history[-100:]):.3f}")

        final_avg_score = total_score / episodes
        print(f"\nRésultats finaux après {episodes} épisodes:")
        print(f"Score moyen final: {final_avg_score:.3f}")
        print(f"Taux de victoire final: {win_count / episodes:.2%}")
        print(f"Taux de match nul final: {draw_count / episodes:.2%}")

        return history


def play_game_mcts(agent: MCTS, env: TicTacToe, display=True):
    """Joue une partie complète avec affichage optionnel."""
    env.reset()
    if display:
        print("Début de la partie!")
        env.display()

    while not env.is_game_over():
        if env._player == 0:  # Tour du MCTS
            action = agent.select_action(env)
            if action == -1:
                break
            if display:
                print(f"\nMCTS choisit l'action: {action}")
        else:
            valid_actions = env.available_actions_ids()
            action = np.random.choice(valid_actions)
            if display:
                print(f"\nJoueur aléatoire choisit l'action: {action}")

        env.step(action)
        env.display()

    if display:
        print("\nPartie terminée!")
        if env.score() > 0:
            print("MCTS gagne!")
        elif env.score() < 0:
            print("Joueur aléatoire gagne!")
        else:
            print("Match nul!")

    return env.score()


# Dans la partie principale (main), ajouter :
if __name__ == "__main__":
    env = TicTacToe()
    mcts_agent = MCTS(n_simulations=100, C=sqrt(2))

    print("Début de l'entraînement MCTS...")
    history = mcts_agent.train(env, episodes=10000)

    print("\nJouer des parties de démonstration...")
    demo_scores = []
    for episode in tqdm(range(100), desc="Playing demo games"):
        score = play_game_mcts(mcts_agent, env, display=True)
        demo_scores.append(score)

    print(f"\nScore moyen sur les parties de démonstration: {np.mean(demo_scores):.3f}")
