import csv
import json
import os
import h5py
import numpy as np
from tqdm import tqdm
import copy
from environment.FarkelEnv import FarkleDQNEnv
from environment.tictactoe import TicTacToe


class RandomRolloutAgent:
    def __init__(self, n_simulations=100, gamma=0.99, max_depth=100):
        self.n_simulations = n_simulations
        self.gamma = gamma
        self.max_depth = max_depth

    def select_action(self, env):
        """Sélectionne la meilleure action basée sur les simulations."""
        valid_actions = env.available_actions_ids()

        if len(valid_actions) == 0:  # Correction ici
            return None

        action_scores = {}

        # Évaluer chaque action possible
        for action in valid_actions:
            total_score = 0

            # Faire plusieurs simulations pour cette action
            for _ in range(self.n_simulations):
                score = self.simulate_game(env, action)
                total_score += score

            # Calculer le score moyen pour cette action
            action_scores[action] = total_score / self.n_simulations

        # Retourner l'action avec le meilleur score
        return max(action_scores.items(), key=lambda x: x[1])[0]

    def simulate_game(self, env, initial_action):
        """Simule une partie à partir d'une action initiale."""
        sim_env = copy.deepcopy(env)

        # Faire l'action initiale
        sim_env.step(initial_action)
        depth = 0

        # Continuer avec des actions aléatoires jusqu'à la fin
        while not sim_env.is_game_over() and depth < self.max_depth:
            possible_actions = sim_env.available_actions_ids()
            if len(possible_actions) == 0:  # Correction ici
                break

            random_action = np.random.choice(possible_actions)
            sim_env.step(random_action)
            depth += 1

        return sim_env.score()

    def train(self, env, episodes: int = 10000, save_path="models/Randomrollout/Farkel/"):
        """Entraîne l'agent sur plusieurs épisodes et sauvegarde les résultats."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        history = []
        win_count = 0
        draw_count = 0

        # Initialisation du fichier CSV avec les en-têtes
        csv_path = os.path.join(save_path, "evaluation_results.csv")
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Episode", "Average Reward", "Average Length"])

        for episode in tqdm(range(episodes), desc="Entraînement RandomRollout"):
            env.reset()
            game_over = False

            while not game_over:
                action = self.select_action(env)
                if action is None:  # Si aucune action valide n'est disponible
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

            # Sauvegarde périodique tous les 1000 épisodes
            if (episode + 1) % 100 == 0 or (episode + 1) == episodes:
                save_file = os.path.join(save_path, f"random_rollout_episode_Farkel_{episode + 1}.h5")
                with h5py.File(save_file, "w") as f:
                    f.create_dataset("episode", data=episode + 1)
                    f.create_dataset("win_rate", data=win_count / (episode + 1))  # Calcul du win_rate à ce moment
                    f.create_dataset("draw_rate", data=draw_count / (episode + 1))  # Calcul du draw_rate à ce moment
                    f.create_dataset("score_history", data=np.array(history))
                print(f"Résultats sauvegardés dans {save_file}")

                # Évalue la politique et sauvegarde dans le fichier CSV
                evaluation_results = self.evaluate_policy(env, episodes=100)
                avg_reward = evaluation_results["average_reward"]
                avg_length = evaluation_results["average_length"]

                # Ajout des résultats au CSV
                with open(csv_path, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([episode + 1, avg_reward, avg_length])
                    csv_file.flush()

        return history

    def evaluate_policy(self, env, episodes=10):
        """Évalue la politique de l'agent."""
        total_rewards = []
        total_lengths = []

        for _ in tqdm(range(episodes), desc="Évaluation de la politique"):
            env.reset()
            state = env.state_description()

            rewards = 0
            length = 0

            while not env.is_game_over():
                action = self.select_action(env)
                if action is None:
                    break

                env.step(action)
                rewards += env.score()  # Cumule les récompenses
                state = env.state_description()
                length += 1

            total_rewards.append(rewards)
            total_lengths.append(length)

        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(total_lengths)

        print("\nÉvaluation terminée")
        print(f"Récompense moyenne : {avg_reward:.3f}")
        print(f"Longueur moyenne des parties : {avg_length:.3f}")

        return {
            "average_reward": avg_reward,
            "average_length": avg_length,
        }


def play_game(agent: RandomRolloutAgent, env, display=True):
    """Joue une partie complète avec affichage optionnel."""
    env.reset()
    if display:
        print("Début de la partie!")
        env.display()

    while not env.is_game_over():
        if env._player == 0:  # Tour de RandomRollout
            action = agent.select_action(env)
            if action is None:
                break
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
            print("Joueur aléatoire gagne!")
        else:
            print("Match nul!")

    return env.score()


if __name__ == "__main__":
    env = TicTacToe()
    env = FarkleDQNEnv(num_players=2,target_score=2000)
    agent = RandomRolloutAgent(n_simulations=50, gamma=0.95)
    history = agent.train(env, episodes=1000)
