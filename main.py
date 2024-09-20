import sys
import os
from environment.tictactoe import TicTacToe
from outils import human_move, play
from rand.random import random_agent, random_agent_line_world, random_agent_grid_world

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.line_word import LineWorld
from environment.grid_word import GridWorld
from environment.tictactoe import TicTacToe
import rand
from outils import human_move_line_world, play_line_world, human_move_grid_world, play_grid_world

def choose_single_player_agent(game_name):
    """
    Fonction qui permet à l'utilisateur de choisir s'il veut jouer en tant qu'humain ou laisser un agent aléatoire jouer.
    """
    choice = input(f"Choisissez l'agent pour {game_name} (1: Humain, 2: Random) : ")
    if choice == '1':
        return 'human'
    elif choice == '2':
        return 'random'
    else:
        print("Choix invalide, par défaut l'agent sera Random.")
        return 'random'

if __name__ == "__main__":
    # Choix du jeu
    game_choice = input("Choisissez le jeu (1: LineWorld, 2: GridWorld) : ")

    if game_choice == '1':
        # Initialiser l'environnement LineWorld avec une longueur de 5
        line_world = LineWorld(length=5)

        # Choisir pour un seul joueur
        player_choice = choose_single_player_agent("LineWorld")

        if player_choice == 'human':
            # Jouer à LineWorld avec un humain contre un agent random
            play_line_world(line_world, human_move_line_world, random_agent_line_world)
        else:
            # Jouer à LineWorld avec deux agents random
            play_line_world(line_world, random_agent_line_world, random_agent_line_world)

    elif game_choice == '2':
        # Initialiser l'environnement GridWorld avec une grille de 5x5
        grid_world = GridWorld(width=5, height=5)

        # Choisir pour un seul joueur
        player_choice = choose_single_player_agent("GridWorld")

        if player_choice == 'human':
            # Jouer à GridWorld avec un humain contre un agent random
            play_grid_world(grid_world, human_move_grid_world, random_agent_grid_world)
        else:
            # Jouer à GridWorld avec deux agents random
            play_grid_world(grid_world, random_agent_grid_world, random_agent_grid_world)

    else:
        print("Choix invalide. Veuillez entrer 1 ou 2.")


if __name__ == "__main__":
    t = TicTacToe()
    play(t, human_move, random_agent, print_game=True)  # Utilisation de random_agent pour le joueur O