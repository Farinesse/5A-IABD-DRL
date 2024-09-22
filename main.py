import sys
import os
import numpy as np
from environment.tictactoe_new import TicTacToe_new
from environment.line_word_new import LineWorld
from environment.grid_word_new import GridWorld
from outils import human_move, play, human_move_line_world, play_line_world, human_move_grid_world, play_grid_world
from rand.random import random_agent, random_agent_line_world, random_agent_grid_world, play_with_q_agent
from QLearning.qlearning import tabular_q_learning   # Assurez-vous que la fonction tabular_q_learning est définie

# Ajouter le chemin absolu au système
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def choose_single_player_agent(game_name):
    """
    Fonction qui permet à l'utilisateur de choisir s'il veut jouer en tant qu'humain, avec un agent aléatoire ou un agent Q-Learning.
    """
    choice = input(f"Choisissez l'agent pour {game_name} (1: Humain, 2: Random, 3: Q-Learning) : ")
    if choice == '1':
        return 'human'
    elif choice == '2':
        return 'random'
    elif choice == '3':
        return 'q_learning'
    else:
        print("Choix invalide, par défaut l'agent sera Random.")
        return 'random'



if __name__ == "__main__":
    # Choix du jeu
    game_choice = input("Choisissez le jeu (1: LineWorld, 2: GridWorld, 3: TicTacToe) : ")

    if game_choice == '1':
        # Initialiser l'environnement LineWorld avec une longueur de 5
        line_world = LineWorld(length=5)

        # Choisir pour un seul joueur
        player_choice = choose_single_player_agent("LineWorld")

        if player_choice == 'human':
            # Jouer à LineWorld avec un humain contre un agent random
            play_line_world(line_world, human_move_line_world, random_agent_line_world)
        elif player_choice == 'q_learning':
            # Jouer à LineWorld avec un agent Q-Learning contre un agent random
            play_with_q_agent(LineWorld, random_agent_line_world, 5)  # Passer la longueur
        else:
            # Jouer à LineWorld avec deux agents random
            play_line_world(line_world, random_agent_line_world, random_agent_line_world)

    elif game_choice == '2':
        # Choisir pour un seul joueur
        player_choice = choose_single_player_agent("GridWorld")

        if player_choice == 'human':
            # Jouer à GridWorld avec un humain contre un agent random
            grid_world = GridWorld(width=5, height=5)
            play_grid_world(grid_world, human_move_grid_world, random_agent_grid_world)
        elif player_choice == 'q_learning':
            # Jouer à GridWorld avec un agent Q-Learning contre un agent random
            play_with_q_agent(GridWorld, random_agent_grid_world, 5, 5)  # Passer width=5 et height=5
        else:
            # Jouer à GridWorld avec deux agents random
            grid_world = GridWorld(width=5, height=5)
            play_grid_world(grid_world, random_agent_grid_world, random_agent_grid_world)

    elif game_choice == '3':
        # Initialiser l'environnement TicTacToe
        tic_tac_toe = TicTacToe_new()

        # Choisir pour un seul joueur
        player_choice = choose_single_player_agent("TicTacToe")

        if player_choice == 'human':
            # Jouer à TicTacToe avec un humain contre un agent random
            play(tic_tac_toe, human_move, random_agent)
        elif player_choice == 'q_learning':
            # Jouer à TicTacToe avec un agent Q-Learning contre un agent random
            play_with_q_agent(TicTacToe_new, random_agent)
        else:
            # Jouer à TicTacToe avec deux agents random
            play(tic_tac_toe, random_agent, random_agent)

    else:
        print("Choix invalide. Veuillez entrer 1, 2 ou 3.")
