import sys
import os
import numpy as np
from tensorflow import keras
import time
import tensorflow as tf
from GUI.Farkel_GUI import main_gui
from algos.DQN.ddqn import double_dqn_no_replay
# Importer les modules nécessaires
from environment.FarkelEnv import FarkleEnv, FarkleDQNEnv
from environment.tictactoe import TicTacToe
from environment.line_word import LineWorld
from environment.grid_word import GridWorld
from functions.outils import human_move, play, human_move_line_world, play_line_world, human_move_grid_world, play_grid_world
from functions.random import random_agent_line_world, random_agent_grid_world, play_with_q_agent, \
    play_dqn_vs_random, play_with_dqn

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
        tic_tac_toe = TicTacToe()

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
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(output_dim)
            ])

            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

            # Entraîner l'agent Deep Q-Learning
            '''trained_model = deep_q_learning(
                model=model,
                target_model=target_model,
                env=tic_tac_toe,
                num_episodes=20000,
                gamma=0.99,
                alpha=0.0005,
                start_epsilon=1.0,
                end_epsilon=0.01, #
                memory_size=5000,
                batch_size=256,
                update_target_steps=1000
            )'''
            final_online_model, final_target_model = double_dqn_no_replay(
                online_model=model,
                target_model=model,
                env=tic_tac_toe,
                num_episodes=100000,
                gamma=0.99,
                alpha=0.001,
                start_epsilon=1,
                end_epsilon=0.001,
                update_target_steps=1000,
                save_path='double_dqn_tictactoe_final_test.h5'
            )
            '''  final_online_model, _ = double_dqn_with_replay(
                online_model=model,
                target_model=model,
                env=tic_tac_toe,
                num_episodes=50000,
                gamma=0.99,
                alpha=0.001,
                start_epsilon=1.0,
                end_epsilon=0.01,
                update_target_steps=100,
                batch_size=64,
                memory_size=1024,
                save_path='models/TEST3_double_dqn_exp_replay_tictactoe_final.h5'
            )'''

            # Jouer une partie avec l'agent DQN contre un agent random
            play_dqn_vs_random(tic_tac_toe, final_online_model, random_agent_func=random_agent, episodes=100)

            # Calculer le nombre de parties par seconde
            calculate_games_per_second(tic_tac_toe, final_online_model, random_agent)
        else:
            # Jouer à TicTacToe avec deux agents random
            play(tic_tac_toe, random_agent, random_agent)
    elif game_choice == '4':
        # Farkel
        print("Choisissez le mode de jeu Farkel :")
        print("1: Humain vs Random")
        print("2: random vs Agent")
        print("3: Entraîner l'agent")

        farkel_choice = input("Votre choix : ")
        if farkel_choice == '1':
            players = int(input("number of players : "))
            main_gui(players)
        elif farkel_choice == '2':
            print("Entraînement de l'agent Q-learning...")
            env = FarkleDQNEnv()
            model_path = "models/models/double_dqn_model_Farkel_test1"
            model = keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
            play_with_dqn(env, model, random_agent=None, episodes=10)

        #trained_agent, _ = train(episodes=100)

        elif farkel_choice == '3':

            print("Entraînement de l'agent Q-learning...")
            agent = FarkleDQNEnv()
            trained_agent, trained_model = double_dqn_no_replay(episodes=1000)
            print(f"Évaluation - Score moyen : , Écart-type : ")

        else:
            print("Choix invalide. Retour au menu principal.")

    else:
        print("Choix invalide. Veuillez entrer 1, 2, 3 ou 4.")
