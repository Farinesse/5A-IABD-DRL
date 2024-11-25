import numpy as np
from tensorflow import keras
from tqdm import tqdm
import csv
from datetime import datetime
import os
import tensorflow as tf

# Environnements
from environment.FarkelEnv import FarkleDQNEnv
from environment.tictactoe import TicTacToe
from environment.grid_word import GridWorld
from environment.line_word import LineWorld

# Algorithmes DQN
from algos.DQN.ddqn import double_dqn_no_replay
from algos.DQN.ddqn_exp_replay import double_dqn_with_replay
from algos.DQN.deep_qlearning import deep_q_learning

# Algorithmes Policy Gradient
from algos.PolicyGradientMethods.reinforce import REINFORCE
from algos.PolicyGradientMethods.ppo import PPO_A2C_Style
from algos.PolicyGradientMethods.one_step_actor_critic import REINFORCEWithCritic
from algos.PolicyGradientMethods.reinforce_meanbase import REINFORCEWithBaseline


def setup_directories():
    """Crée les répertoires nécessaires."""
    directories = ['models', 'results', 'checkpoints_actor', 'checkpoints_critic']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_results(results, algorithm, env_name):
    """Sauvegarde les résultats dans un fichier CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/training_results_{env_name}_{algorithm}_{timestamp}.csv"

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Win_Rate'])
        for row in results:
            writer.writerow(row)


def get_env_dimensions(env_name):
    """Retourne les dimensions d'entrée/sortie pour chaque environnement."""
    dims = {
        'farkle': (12, 128),
        'tictactoe': (27, 9),
        'gridworld': (2, 4),
        'lineworld': (1, 2)
    }
    return dims.get(env_name)


def create_environment(env_name, **kwargs):
    """Crée l'environnement approprié."""
    env_creators = {
        'farkle': lambda: FarkleDQNEnv(target_score=kwargs.get('target_score', 2000)),
        'tictactoe': TicTacToe,
        'gridworld': lambda: GridWorld(width=kwargs.get('width', 5), height=kwargs.get('height', 5)),
        'lineworld': lambda: LineWorld(length=kwargs.get('length', 5))
    }
    return env_creators[env_name]()


if __name__ == "__main__":
    # Configuration initiale
    setup_directories()
    tf.get_logger().setLevel('ERROR')

    print("Choisissez l'environnement :")
    print("1. Farkle")
    print("2. TicTacToe")
    print("3. GridWorld")
    print("4. LineWorld")

    env_choice = input("Votre choix d'environnement (1-4) : ")

    env_mapping = {
        '1': 'farkle',
        '2': 'tictactoe',
        '3': 'gridworld',
        '4': 'lineworld'
    }

    env_name = env_mapping.get(env_choice)
    if not env_name:
        print("Choix invalide!")
        exit()

    # Configuration de l'environnement
    if env_name == 'farkle':
        target_score = int(input("Score cible (défaut: 2000): ") or "2000")
        env = create_environment(env_name, target_score=target_score)
    else:
        env = create_environment(env_name)

    input_dim, output_dim = get_env_dimensions(env_name)

    print("\nChoisissez l'algorithme :")
    print("=== DQN ===")
    print("1. DQN classique")
    print("2. Double DQN sans replay")
    print("3. Double DQN avec replay")
    print("=== Policy Gradient ===")
    print("4. REINFORCE")
    print("5. REINFORCE with Baseline")
    print("6. PPO (A2C Style)")
    print("7. REINFORCE with Critic (Actor-Critic)")

    algo_choice = input("Votre choix (1-7) : ")

    # Configuration et exécution de l'algorithme choisi
    if algo_choice in ['1', '2', '3']:  # Algorithmes DQN
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=input_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(output_dim)
        ])
        target_model = keras.models.clone_model(model)
        target_model.set_weights(model.get_weights())

        if algo_choice == '1':
            print("\nEntraînement DQN classique...")
            trained_model = deep_q_learning(
                model=model,
                target_model=target_model,
                env=env,
                num_episodes=10,
                gamma=0.99,
                alpha=0.001,
                start_epsilon=1.0,
                end_epsilon=0.01,
                memory_size=512,
                batch_size=128,
                update_target_steps=50
            )

        elif algo_choice == '2':
            print("\nEntraînement Double DQN sans replay...")
            trained_model, _ = double_dqn_no_replay(
                online_model=model,
                target_model=target_model,
                env=env,
                num_episodes=50,
                gamma=0.99,
                alpha=0.0001,
                start_epsilon=1.0,
                end_epsilon=0.0001,
                update_target_steps=100,
                save_path=f"models/ddqn_{env_name}"
            )

        elif algo_choice == '3':
            print("\nEntraînement Double DQN avec replay...")
            trained_model, _ = double_dqn_with_replay(
                online_model=model,
                target_model=target_model,
                env=env,
                num_episodes=100000,
                gamma=0.99,
                alpha=0.0001,
                start_epsilon=1.0,
                end_epsilon=0.01,
                update_target_steps=100,
                batch_size=32,
                memory_size=128,
                save_path=f'models/ddqn_replay_{env_name}'
            )

    else:  # Algorithmes Policy Gradient
        if algo_choice == '4':
            print("\nEntraînement REINFORCE...")
            agent = REINFORCE(
                state_dim=input_dim,
                action_dim=output_dim,
                alpha=0.001,
                gamma=0.99
            )
            history = agent.train(env, episodes=20000)
            agent.save(f"models/reinforce_{env_name}")

        elif algo_choice == '5':
            print("\nEntraînement REINFORCE with Baseline...")
            agent = REINFORCEWithBaseline(
                state_dim=input_dim,
                action_dim=output_dim,
                alpha_theta=0.00005,
                alpha_w=0.0005,
                gamma=0.995
            )
            history = agent.train(
                env,
                episodes=20000,
                eval_frequency=1000,
                eval_episodes=100
            )
            agent.plot_metrics(f"results/:{env_name}/evaluation_metrics_reinforce.csv")

        elif algo_choice == '6':
            print("\nEntraînement PPO...")
            agent = PPO_A2C_Style(
                state_dim=input_dim,
                action_dim=output_dim,
                alpha=0.0001,
                gamma=0.99,
                clip_ratio=0.1,
                epsilon_decay_episodes=5000
            )
            history = agent.train(env, episodes=10000)
            agent.save(f"models/results/:{env_name}/ppo_{env_name}")

        elif algo_choice == '7':
            print("\nEntraînement Actor-Critic...")
            agent = REINFORCEWithCritic(
                state_dim=input_dim,
                action_dim=output_dim,
                alpha_policy=0.0001,
                alpha_critic=0.001,
                gamma=0.99
            )
            history = agent.train(env, episodes=100000)
            agent.save(f"models/results/:{env_name}/actor_critic_{env_name}")

    print("\nEntraînement terminé!")