import numpy as np
from tensorflow import keras
from tqdm import tqdm
import csv
from datetime import datetime
import os
import tensorflow as tf
from GUI.Menu_GUI import TrainingGUI  # Import du menu GUI
import tkinter as tk

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
    # Initialiser les répertoires nécessaires
    setup_directories()
    tf.get_logger().setLevel('ERROR')

    # Lancer le menu GUI
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()
