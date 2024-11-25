import json
import os
import h5py
import numpy as np
from tqdm import tqdm
import copy
import csv
from datetime import datetime


class RandomRolloutAgent:
    def __init__(self, n_simulations=100, gamma=0.99):
        self.n_simulations = n_simulations
        self.gamma = gamma

    def select_action(self, env):
        valid_actions = env.available_actions_ids()
        if len(valid_actions) == 0:
            return None

        action_scores = {}

        # Pour chaque action possible, faire des simulations
        for action in valid_actions:
            total_score = 0

            # Faire plusieurs simulations pour cette action
            for _ in range(self.n_simulations):
                score = self.simulate_game(env, action)
                total_score += score

            action_scores[action] = total_score / self.n_simulations

        return max(action_scores.items(), key=lambda x: x[1])[0]

    def simulate_game(self, env, initial_action):
        sim_env = copy.deepcopy(env)
        sim_env.step(initial_action)

        # Jouer jusqu'à la fin avec des actions aléatoires
        while not sim_env.is_game_over():
            possible_actions = sim_env.available_actions_ids()
            if len(possible_actions) == 0:
                break

            random_action = np.random.choice(possible_actions)
            sim_env.step(random_action)

        return sim_env.score()

    def train(self, env, episodes=1000, eval_interval=100, save_dir="results/random_rollout/"):
        """
        Entraîne l'agent avec évaluation périodique
        """
        # Création du dossier de sauvegarde
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"run_{timestamp}")
        os.makedirs(save_path, exist_ok=True)

        # Fichier CSV pour les métriques
        csv_path = os.path.join(save_path, "metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Win_Rate', 'Draw_Rate', 'Avg_Score', 'Eval_Win_Rate', 'Eval_Avg_Score'])

        stats = []
        history = []
        win_count = 0
        draw_count = 0

        for episode in tqdm(range(episodes), desc="Training"):
            env.reset()
            episode_done = False

            while not episode_done:
                action = self.select_action(env)
                if action is None:
                    break

                env.step(action)
                episode_done = env.is_game_over()

            score = env.score()
            history.append(score)

            if score > 0:
                win_count += 1
            elif score == 0:
                draw_count += 1

            # Évaluation périodique
            if (episode + 1) % eval_interval == 0:
                win_rate = win_count / (episode + 1)
                draw_rate = draw_count / (episode + 1)
                avg_score = np.mean(history[-eval_interval:])

                # Faire une évaluation
                eval_stats = self.evaluate(env, n_games=20)

                # Sauvegarder les métriques
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode + 1,
                        win_rate,
                        draw_rate,
                        avg_score,
                        eval_stats['win_rate'],
                        eval_stats['avg_score']
                    ])

                # Afficher les stats
                print(f"\nEpisode {episode + 1}")
                print(f"Training - Win Rate: {win_rate:.2%}, Draw Rate: {draw_rate:.2%}")
                print(f"Evaluation - Win Rate: {eval_stats['win_rate']:.2%}, Avg Score: {eval_stats['avg_score']:.2f}")

                # Sauvegarder un checkpoint
                self.save_checkpoint(save_path, episode + 1, {
                    'win_rate': win_rate,
                    'draw_rate': draw_rate,
                    'avg_score': avg_score,
                    'eval_stats': eval_stats
                })

        return history

    def evaluate(self, env, n_games=100):
        """
        Évalue l'agent sur plusieurs parties
        """
        wins = 0
        total_score = 0

        for _ in range(n_games):
            env.reset()
            game_done = False

            while not game_done:
                action = self.select_action(env)
                if action is None:
                    break

                env.step(action)
                game_done = env.is_game_over()

            score = env.score()
            total_score += score

            if score > 0:
                wins += 1

        return {
            'win_rate': wins / n_games,
            'avg_score': total_score / n_games
        }

    def save_checkpoint(self, save_dir, episode, stats):
        """
        Sauvegarde un checkpoint de l'agent et ses statistiques
        """
        checkpoint = {
            'episode': episode,
            'n_simulations': self.n_simulations,
            'gamma': self.gamma,
            'stats': stats
        }

        checkpoint_path = os.path.join(save_dir, f"checkpoint_ep_{episode}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=4)


if __name__ == "__main__":
    from environment.FarkelEnv import FarkleDQNEnv

    # Configuration
    env = FarkleDQNEnv(num_players=2, target_score=1000)
    agent = RandomRolloutAgent(n_simulations=20, gamma=0.95)

    # Entraînement avec évaluation
    history = agent.train(
        env,
        episodes=1000,
        eval_interval=100,
        save_dir="results/farkel/random_rollout"
    )