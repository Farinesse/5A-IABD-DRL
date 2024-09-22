import numpy as np

import random
from QLearning.qlearning import tabular_q_learning


def random_agent(game):
    """
    Agent qui choisit une action de manière aléatoire parmi les actions disponibles.

    :param game: Instance du jeu TicTacToe ou autre.
    :return: Action aléatoire choisie.
    """
    return random.choice(game.available_actions_ids())



def random_agent_line_world(game):
    """
    Agent qui choisit une action de manière aléatoire dans LineWorld.
    """
    return random.choice(game.available_actions_ids())

def random_agent_grid_world(game):
    """
    Agent qui choisit une action de manière aléatoire dans GridWorld.
    """
    return random.choice(game.available_actions_ids())


def play_with_q_agent(env_type, random_agent_fn, *env_args):
    """
    Joue un match avec un agent Q-Learning contre un agent aléatoire.
    :param env_type: Le type de l'environnement (doit respecter GameEnv)
    :param random_agent_fn: Fonction qui définit l'agent aléatoire
    :param env_args: Arguments pour initialiser l'environnement
    """
    # Entraînement de l'agent Q-Learning
    print("Entraînement de l'agent Q-Learning...")
    Pi, Q = tabular_q_learning(lambda: env_type(*env_args), alpha=0.1, epsilon=0.1, gamma=0.999, nb_iter=100000)

    # Jouer un match avec la politique apprise contre un agent aléatoire
    print("Jouer un match avec l'agent Q-Learning contre un agent aléatoire...")
    env = env_type(*env_args)
    env.reset()

    while not env.is_game_over():
        state = env.state_id()

        # Agent Q-Learning utilise la politique apprise
        if state in Pi:
            action = Pi[state]
        else:
            action = np.random.choice(env.available_actions_ids())

        env.step(action)  # L'agent Q-Learning joue

        if not env.is_game_over():
            # L'agent aléatoire joue
            random_action = random_agent_fn(env)
            env.step(random_action)

        env.display()  # Affiche l'état du jeu après chaque action

    print("Jeu terminé. Score final:", env.score())
