import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from tqdm import tqdm
import tkinter as tk
from GUI.Farkel_GUI import FarkleGUI, main_gui
from QLearning.Fakel_dql import train, evaluate, FarkleDQNAgent
# Importer les modules nécessaires
from QLearning.deep_qlearning import deep_q_learning
from environment.FarkelEnv import FarkleEnv
from environment.tictactoe_new import TicTacToe_new
from environment.line_word_new import LineWorld
from environment.grid_word_new import GridWorld
from outils import human_move, play, human_move_line_world, play_line_world, human_move_grid_world, play_grid_world
from rand.random import random_agent, random_agent_line_world, random_agent_grid_world, play_with_q_agent, \
    farkel_random_player, \
    play_with_dqn, play_dqn_vs_random, play_farkel_human_vs_random

# Ajouter le chemin absolu au système
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fonction pour choisir l'agent avec lequel jouer
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

# Fonction pour encoder l'état du jeu
def encode_state(board):
    """
    Encode l'état du jeu en un vecteur. Par exemple, pour TicTacToe, un vecteur de 9 éléments est généré.
    """
    return np.array(board.flatten())

# Fonction pour l'agent aléatoire
def random_agent(env):
    # Sélectionne une action aléatoire parmi les actions disponibles
    return np.random.choice(env.available_actions_ids())

# Calculer le nombre de parties par seconde
def calculate_games_per_second(env, model, random_agent_func, num_games=1000):
    start_time = time.time()
    play_dqn_vs_random(env, model, random_agent_func=random_agent_func, episodes=num_games)
    end_time = time.time()
    elapsed_time = end_time - start_time
    games_per_second = num_games / elapsed_time
    print(f"Nombre de parties par seconde : {games_per_second}")
    return games_per_second

if __name__ == "__main__":
    # Choix du jeu
    game_choice = input("Choisissez le jeu (1: LineWorld, 2: GridWorld, 3: TicTacToe, 4: Farkel(GUI) : ")

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
            # Initialiser les dimensions pour le modèle DQN
            input_dim = 27  # TicTacToe a 9 cases
            output_dim = 9  # 9 actions possibles

            # Créer le modèle principal et le modèle cible avec normalisation batch
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(input_dim,)),
                keras.layers.Dense(128, activation='relu'),  # Less neurons since the game is simple
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(output_dim)  # No activation on output layer
            ])

            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

            # Entraîner l'agent Deep Q-Learning
            trained_model = deep_q_learning(
                model=model,
                target_model=target_model,
                env=tic_tac_toe,
                num_episodes=2000,
                gamma=0.95,
                alpha=0.0001,
                start_epsilon=1.0,
                end_epsilon=0.01,  #
                memory_size=5000,
                batch_size=128  #
            )

            # Jouer une partie avec l'agent DQN contre un agent random
            play_dqn_vs_random(tic_tac_toe, trained_model, random_agent_func=random_agent, episodes=100)




            # Calculer le nombre de parties par seconde
            calculate_games_per_second(tic_tac_toe, trained_model, random_agent)
        else:
            # Jouer à TicTacToe avec deux agents random
            play(tic_tac_toe, random_agent, random_agent)
    elif game_choice == '4':
        # Farkel
        print("Choisissez le mode de jeu Farkel :")
        print("1: Humain vs Humain")
        print("2: Humain vs Random")
        print("3: Humain vs Q-learning")
        print("4: Entraîner l'agent Q-learning")

        farkel_choice = input("Votre choix : ")
        if farkel_choice == '1':
            players = int(input("number of players : "))
            main_gui(players)
        elif farkel_choice == '2':

            env = FarkleEnv(num_players=2)
            root = tk.Tk()
            gui = FarkleGUI(root, env)
            play_farkel_human_vs_random(env, gui, root)
            root.mainloop()

        elif farkel_choice == '3':

            print("Entraînement de l'agent Q-learning...")
            env = FarkleEnv()
            agent = FarkleDQNAgent(env)
            trained_agent, _ = train(episodes=100)

            #play_farkel_human_vs_qlearning(trained_agent)
        elif farkel_choice == '4':

            print("Entraînement de l'agent Q-learning...")
            env = FarkleEnv()
            agent = FarkleDQNAgent(env)
            trained_agent, trained_model = train(episodes=1000)
            mean_score, std_score = evaluate(trained_agent)
            print(f"Évaluation - Score moyen : {mean_score:.2f}, Écart-type : {std_score:.2f}")

        else:
            print("Choix invalide. Retour au menu principal.")

    else:
        print("Choix invalide. Veuillez entrer 1, 2, 3 ou 4.")
