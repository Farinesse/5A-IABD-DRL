import random

def random_agent(game):
    """
    Agent qui choisit une action de manière aléatoire parmi les actions disponibles.

    :param game: Instance du jeu TicTacToe ou autre.
    :return: Action aléatoire choisie.
    """
    return random.choice(game.available_actions())



def random_agent_line_world(game):
    """
    Agent qui choisit une action de manière aléatoire dans LineWorld.
    """
    return random.choice(game.available_actions())

def random_agent_grid_world(game):
    """
    Agent qui choisit une action de manière aléatoire dans GridWorld.
    """
    return random.choice(game.available_actions())