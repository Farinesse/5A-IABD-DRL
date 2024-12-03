import os
import pickle
import numpy as np
from tqdm import tqdm

from environment.FarkelEnv import FarkleDQNEnv


class RandomRolloutAgent:
    def __init__(self, n_simulations=100, gamma=0.99):
        self.n_simulations = n_simulations
        self.gamma = gamma
        self.actions_taken = []

    def _simulate_game(self, env, start_action: int) -> float:
        """Simule une partie complète à partir d'un état et d'une action initiale."""
        sim_env = env.copy()
        try:
            sim_env.step(start_action)
            if sim_env.is_game_over():
                return sim_env.score()
        except ValueError:
            return float('-inf')

        while not sim_env.is_game_over():
            valid_actions = sim_env.available_actions_ids()
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            sim_env.step(action)

        return sim_env.score()

    def select_action(self, env) -> int:
        """Sélectionne la meilleure action basée sur les simulations Monte Carlo."""
        valid_actions = env.available_actions_ids()
        if len(valid_actions) == 0:
            return -1

        action_scores = np.zeros(len(valid_actions))

        for i, action in enumerate(valid_actions):
            scores = []
            for _ in range(self.n_simulations):
                score = self._simulate_game(env, action)
                scores.append(score)
            action_scores[i] = np.mean(scores)

        best_action_idx = np.argmax(action_scores)
        self.actions_taken.append(best_action_idx)
        return valid_actions[best_action_idx]

    def train(self, env, episodes: int = 1000, save_metrics_path=None, save_model_path=None):
        """Entraîne l'agent sur plusieurs épisodes et enregistre les métriques."""
        history = []
        win_count = 0
        draw_count = 0
        total_score = 0
        metrics = []

        for episode in tqdm(range(episodes), desc="Entraînement RandomRollout"):
            env.reset()
            game_over = False

            while not game_over:
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

            if (episode + 1) % 100 == 0:
                win_rate = win_count / (episode + 1)
                draw_rate = draw_count / (episode + 1)
                avg_score = total_score / (episode + 1)
                print(f"\nÉpisode {episode + 1}")
                print(f"Taux de victoire: {win_rate:.2%}")
                print(f"Taux de match nul: {draw_rate:.2%}")
                print(f"Score moyen global: {avg_score:.3f}")
                print(f"Score moyen sur les derniers 100 épisodes: {np.mean(history[-100:]):.3f}")

            if save_metrics_path and (episode + 1) % 100 == 0:
                metrics.append([episode + 1, win_rate, draw_rate, avg_score, np.mean(history[-100:])])

        if save_metrics_path:
            import pandas as pd
            metrics_df = pd.DataFrame(metrics, columns=["Episode", "Win Rate", "Draw Rate", "Avg Score", "Avg Score (Last 100)"])
            metrics_df.to_csv(save_metrics_path, index=False)

        if save_model_path:
            self.save(save_model_path)

        return history

    def save(self, filepath: str):
        """Sauvegarde le modèle RandomRollout et les actions prises"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_state = {
            'n_simulations': self.n_simulations,
            'gamma': self.gamma,
            'actions_taken': self.actions_taken
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

        print(f"Modèle et actions sauvegardés dans {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Charge un modèle RandomRollout sauvegardé"""
        # 1. Lecture du fichier binaire
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)

        # 2. Création d'une nouvelle instance de l'agent
        agent = cls(n_simulations=model_state['n_simulations'], gamma=model_state['gamma'])

        # 3. Restauration des actions prises
        agent.actions_taken = model_state['actions_taken']
        return agent


def play_game(agent: RandomRolloutAgent, env, display=True):
    """Joue une partie complète avec affichage optionnel."""
    env.reset()
    if display:
        print("Début de la partie!")
        env.display()

    while not env.is_game_over():
        action = agent.select_action(env)
        if display:
            print(f"\nRandomRollout choisit l'action: {action}")
        env.step(action)
        if display:
            env.display()

    if display:
        print("\nPartie terminée!")
        if env.score() > 0:
            print("RandomRollout gagne!")
        elif env.score() < 0:
            print("Adversaire gagne!")
        else:
            print("Match nul!")

    return env.score()


if __name__ == "__main__":
    # env = GridWorld(width=5, height=5)
    env = FarkleDQNEnv(target_score=5000)
    # env = TicTacToe()  # Exemple avec TicTacToe
    agent = RandomRolloutAgent(n_simulations=10, gamma=0.99)

    save_metrics_path = "metrics_random_rollout.csv"  # Sauvegarde des métriques
    save_model_path = "models/random_rollout_model.pkl"  # Sauvegarde du modèle

    history = agent.train(env, episodes=1000, save_metrics_path=save_metrics_path, save_model_path=save_model_path)

    play_game(agent, env, display=True)
