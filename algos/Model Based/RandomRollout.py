import numpy as np
from typing import List, Tuple
from tqdm import tqdm

from environment.tictactoe import TicTacToe


class RandomRolloutAgent:
    def __init__(self, n_simulations=100, gamma=0.99):
        self.n_simulations = n_simulations
        self.gamma = gamma

    def _simulate_game(self, env, start_action: int) -> float:
        """Simule une partie complète à partir d'un état et d'une action initiale."""
        # Copie de l'environnement pour la simulation
        sim_env = TicTacToe()
        sim_env._board = env._board.copy()
        sim_env._player = env._player
        sim_env._is_game_over = env._is_game_over
        sim_env._score = env._score

        # Joue l'action initiale
        try:
            sim_env.step(start_action)
            if sim_env.is_game_over():
                return sim_env.score()
        except ValueError:
            return float('-inf')

        # Continue la simulation avec des actions aléatoires jusqu'à la fin
        while not sim_env.is_game_over():
            valid_actions = sim_env.available_actions_ids()
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            sim_env.step(action)

        return sim_env.score()

    def select_action(self, env: TicTacToe) -> int:
        """Sélectionne la meilleure action basée sur les simulations Monte Carlo."""
        valid_actions = env.available_actions_ids()
        if len(valid_actions) == 0:
            return -1

        action_scores = np.zeros(len(valid_actions))

        # Effectue plusieurs simulations pour chaque action valide
        for i, action in enumerate(valid_actions):
            scores = []
            for _ in range(self.n_simulations):
                score = self._simulate_game(env, action)
                scores.append(score)
            action_scores[i] = np.mean(scores)

        # Sélectionne l'action avec le meilleur score moyen
        best_action_idx = np.argmax(action_scores)
        return valid_actions[best_action_idx]

    def train(self, env: TicTacToe, episodes: int = 1000):
        """Évalue les performances de l'agent sur plusieurs épisodes."""
        history = []
        win_count = 0
        draw_count = 0

        for episode in tqdm(range(episodes), desc="Évaluation RandomRollout"):
            env.reset()
            game_over = False

            while not game_over:
                if env._player == 0:  # Tour de RandomRollout
                    action = self.select_action(env)
                    if action == -1:
                        break
                    env.step(action)
                game_over = env.is_game_over()

            # Enregistre le résultat
            score = env.score()
            history.append(score)

            if score > 0:
                win_count += 1
            elif score == 0:
                draw_count += 1

            # Affiche les statistiques tous les 100 épisodes
            if (episode + 1) % 100 == 0:
                win_rate = win_count / (episode + 1)
                draw_rate = draw_count / (episode + 1)
                print(f"\nÉpisode {episode + 1}")
                print(f"Taux de victoire: {win_rate:.2%}")
                print(f"Taux de match nul: {draw_rate:.2%}")
                print(f"Score moyen: {np.mean(history):.3f}")

        return history


def play_game(agent: RandomRolloutAgent, env: TicTacToe, display=True):
    """Joue une partie complète avec affichage optionnel."""
    env.reset()
    if display:
        print("Début de la partie!")
        env.display()

    while not env.is_game_over():
        if env._player == 0:  # Tour de RandomRollout
            action = agent.select_action(env)
            if display:
                print(f"\nRandomRollout choisit l'action: {action}")
        else:
            valid_actions = env.available_actions_ids()
            action = np.random.choice(valid_actions)
            if display:
                print(f"\nJoueur aléatoire choisit l'action: {action}")

        env.step(action)
        if display:
            env.display()

    if display:
        print("\nPartie terminée!")
        if env.score() > 0:
            print("RandomRollout gagne!")
        elif env.score() < 0:
            print("Joueur aléatoire gagne!")
        else:
            print("Match nul!")

    return env.score()


if __name__ == "__main__":
    # Configuration et entraînement
    env = TicTacToe()
    agent = RandomRolloutAgent(n_simulations=100, gamma=0.99)

    # Entraînement/Évaluation sur plusieurs épisodes
    history = agent.train(env, episodes=1000)

    # Joue une partie avec affichage
    play_game(agent, env, display=True)