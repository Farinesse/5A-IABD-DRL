import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from statistics import mean
from algos.DQN.ddqn import epsilon_greedy_action


def logarithmic_decay(episode, start_epsilon, end_epsilon, decay_rate=0.01):
    return max(end_epsilon, start_epsilon - decay_rate * math.log(1 + episode))


def custom_two_phase_decay(episode, start_epsilon, end_epsilon, total_episodes, midpoint=0.5):
    # Point de transition entre phase lente et phase rapide
    transition_point = int(total_episodes * midpoint)

    if episode <= transition_point:
        # Première moitié : décroissance très lente (quasi linéaire)
        progress = episode / transition_point
        return start_epsilon - (start_epsilon - 0.5) * (progress ** 3)
    else:
        # Seconde moitié : décroissance plus rapide (exponentielle)
        remaining_episodes = total_episodes - transition_point
        progress = (episode - transition_point) / remaining_episodes
        return 0.5 * math.exp(-5 * progress)  # 0.5 est la valeur d'epsilon au point de transition


def human_move(game):
    """
    Permet à l'humain de choisir une case sur le plateau (de 0 à 8).
    """
    valid_square = False
    val = None
    while not valid_square:
        square = input('Entrez un choix de case (0-8) : ')
        try:
            val = int(square)
            if val not in game.available_actions_ids():  # Vérifie les actions disponibles
                raise ValueError
            valid_square = True
        except ValueError:
            print("Case invalide, essayez à nouveau.")
    return val


def play(game, player_x, player_o, print_game=True):
    """
    Permet à deux joueurs (humain ou IA) de jouer une partie de TicTacToe.
    """
    if print_game:
        game.display()

        print("state")
        game.state_description()
        print("ACTION")
        game.available_actions_ids()
        print("ACTION MASK")
        game.action_mask()
    letter = 'X'  # Le premier joueur commence avec 'X'
    while not game.is_game_over():
        ''' if letter == 'O':
            print("Joueur O à jouer")
            # Utilisation de l'agent random pour O
            square = player_o(game)
            game.step(square)  # Exécute l'action
        else:
            print("Joueur X à jouer")
            # Utilisation du joueur humain pour X
            #square = player_x(game)
            '''
        square = player_x(game)
        game.step(square)  # Exécute l'action
        print("state : ", game.state_description())
        print("ACTION : ", game.available_actions_ids())

        print("ACTION MASK : ", game.action_mask())

        if print_game:
            print(f'{letter} a joué sur la case {square}')
            game.display()
            print('')

        if game.is_game_over():
            if print_game:
                if game.score() == 1.0:
                    print(f'{letter} a gagné!')
                elif game.score() == 0.0:
                    print('C\'est un match nul!')
            return letter  # Retourne le gagnant ou None en cas de match nul

        # Changement de joueur
        letter = 'O' if letter == 'X' else 'X'


def human_move_line_world(game):
    """
    Permet à l'humain de choisir une action dans LineWorld.
    """
    valid_action = False
    action = None
    while not valid_action:
        print("\nChoisissez une action :")
        print("0: Rester sur place")
        print("1: Aller à gauche")
        print("2: Aller à droite")
        try:
            action = int(input("Votre choix (0, 1, 2) : "))
            if action in game.available_actions_ids():  # Vérifie les actions disponibles
                valid_action = True
            else:
                print("Action invalide ! Veuillez choisir parmi les actions disponibles.")
        except ValueError:
            print("Entrée invalide, veuillez entrer un nombre.")
    return action


def play_line_world(game, player_human, player_random, print_game=True):
    """
    Permet à un humain et un agent aléatoire de jouer à LineWorld.
    """
    if print_game:
        game.display()

    while not game.is_game_over():
        # Tour de l'humain
        action = player_human(game)
        game.step(action)
        if print_game:
            game.display()

        if game.is_game_over():
            if print_game:
                print("Jeu terminé ! L'agent est dans un état terminal.")
            break

        # Tour de l'agent aléatoire
        action = player_random(game)
        game.step(action)
        if print_game:
            game.display()

        if game.is_game_over():
            if print_game:
                print("Jeu terminé ! L'agent est dans un état terminal.")
            break


def human_move_grid_world(game):
    """
    Permet à l'humain de choisir une action dans GridWorld.
    """
    valid_action = False
    action = None
    while not valid_action:
        print("\nChoisissez une action :")
        print("0: Aller en haut")
        print("1: Aller en bas")
        print("2: Aller à gauche")
        print("3: Aller à droite")
        try:
            action = int(input("Votre choix (0, 1, 2, 3) : "))
            if action in game.available_actions_ids():  # Vérifie les actions disponibles
                valid_action = True
            else:
                print("Action invalide ! Veuillez choisir parmi les actions disponibles.")
        except ValueError:
            print("Entrée invalide, veuillez entrer un nombre.")
    return action


def play_grid_world(game, player_human, player_random, print_game=True):
    """
    Permet à un humain et un agent aléatoire de jouer à GridWorld.
    """
    if print_game:
        game.display()

    while not game.is_game_over():
        # Tour de l'humain
        action = player_human(game)
        game.step(action)
        if print_game:
            game.display()

        if game.is_game_over():
            if print_game:
                print("Jeu terminé ! L'agent est dans un état terminal.")
            break

        # Tour de l'agent aléatoire
        action = player_random(game)
        game.step(action)
        if print_game:
            game.display()

        if game.is_game_over():
            if print_game:
                print("Jeu terminé ! L'agent est dans un état terminal.")
            break


def play_game_manual():
    """Fonction pour jouer manuellement contre un adversaire aléatoire."""
    env = None  # FarkleDQNEnv(num_players=2, target_score=5000)
    state, _ = env.reset()
    done = False

    while not env.is_game_over():
        # Affichage plus clair de l'état du jeu
        print("\n" + "=" * 50)
        print("État du jeu:")
        print(f"🎲 Dés actuels: {env.dice_roll}")
        print(f"🎯 Score du tour: {env.round_score}")
        print(f"👥 Scores des joueurs: {env.scores}")
        print(f"🎮 Joueur actuel: {env.current_player + 1}")
        print(f"🎲 Dés restants: {env.remaining_dice}")
        print("=" * 50 + "\n")

        if env.current_player == 0:  # Tour du joueur humain
            # Affichage des actions valides
            print("\nActions valides disponibles:")
            valid_actions = env.available_actions_ids()
            for action_id in valid_actions:
                action_binary = format(action_id, '07b')
                print(f"\nID: {action_id}")
                print(f"Action binaire: {action_binary}")
                # Explication détaillée de l'action
                dice_selection = list(action_binary[:-1])
                stop_action = action_binary[-1]

                # Montrer quels dés seraient gardés
                kept_dice = []
                for i, (die, keep) in enumerate(zip(env.dice_roll, dice_selection)):
                    if keep == '1':
                        kept_dice.append(die)

                print(f"Dés à garder: {kept_dice}")
                print(f"Action stop: {'Oui' if stop_action == '1' else 'Non'}")

            # Saisie de l'action avec validation
            while True:
                try:
                    action = int(input("\nEntrez l'ID de votre action : "))
                    if action_id in valid_actions:
                        #    action = [int(b) for b in format(action_id, '07b')]
                        break
                    print("❌ Action invalide. Veuillez choisir parmi les actions listées.")
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide.")

        else:  # Tour de l'adversaire aléatoire
            print("\nTour de l'adversaire...")
            action = env.get_random_action()
            print(f"L'adversaire choisit : {action}")

        # Exécution de l'action
        state, reward, done, truncated, info = env.step(action)

        # Affichage des résultats
        if info.get("farkle"):
            print(f"\n🎲 FARKLE! Perte de {info['lost_points']} points")
        elif info.get("invalid_action"):
            print("\n❌ Action invalide!")
        elif info.get("stopped"):
            print(f"\n🛑 Tour terminé! Points gagnés: {reward}")
        elif info.get("win"):
            print(f"\n🏆 Victoire! Points finaux: {reward}")
        else:
            print(f"\n✔️ Points gagnés ce coup: {reward}")

    # Affichage des résultats finaux
    print("\n🎮 Partie terminée!")
    print(f"Scores finaux: Joueur 1 = {env.scores[0]}, Joueur 2 = {env.scores[1]}")
    if env.scores[0] > env.scores[1]:
        print("🎉 Félicitations! Vous avez gagné!")
    else:
        print("😔 L'adversaire a gagné. Meilleure chance la prochaine fois!")


def play_with_dqn(env, model, predict_func, episodes=100):
    episode_scores = []
    episode_times = []
    episode_steps = []
    step_times = []
    total_time = 0

    for episode in range(episodes):
        env.reset()
        nb_turns = 0
        start_time = time.time()

        while not env.is_game_over():
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_s = predict_func(model, s_tensor)

            if env.current_player == 0:
                a = epsilon_greedy_action(q_s.numpy(), mask_tensor, env.available_actions_ids(), 0.000001)
                if a not in env.available_actions_ids():
                    # print(f"Action invalide {a}, choix aléatoire à la place.")
                    a = np.random.choice(env.available_actions_ids())
            else:
                a = np.random.choice(env.available_actions_ids())

            env.step(a)
            nb_turns += 1
        end_time = time.time()

        episode_time = end_time - start_time
        episode_scores.append(env.score())
        episode_times.append(episode_time)
        total_time += episode_time
        episode_steps.append(nb_turns)
        step_times.append(episode_time / nb_turns)


    return (
        mean(episode_scores),
        mean(episode_times),
        mean(episode_steps),
        mean(step_times),
        episode_scores.count(1.0) / episodes
    )

def play_with_ddqn(env, policy_net, target_net, predict_func, episodes=100, epsilon=0.1):
    episode_scores = []
    episode_times = []
    episode_steps = []
    step_times = []
    total_time = 0

    for episode in range(episodes):
        env.reset()
        nb_turns = 0
        start_time = time.time()

        while not env.is_game_over():
            # Obtenir l'état actuel
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            # Calcul des Q-values via le réseau principal
            q_s = predict_func(policy_net, s_tensor)

            # Choix de l'action (ε-greedy)
            if env.current_player == 0:
                a = epsilon_greedy_action(q_s.numpy(), mask_tensor, env.available_actions_ids(), epsilon)
                if a not in env.available_actions_ids():
                    a = np.random.choice(env.available_actions_ids())
            else:
                a = np.random.choice(env.available_actions_ids())

            # Exécuter l'action
            env.step(a)
            nb_turns += 1

        # Fin de l'épisode
        end_time = time.time()

        episode_time = end_time - start_time
        episode_scores.append(env.score())
        episode_times.append(episode_time)
        total_time += episode_time
        episode_steps.append(nb_turns)
        step_times.append(episode_time / nb_turns)

    # Calcul des métriques
    return (
        mean(episode_scores),
        mean(episode_times),
        mean(episode_steps),
        mean(step_times),
        episode_scores.count(1.0) / episodes
    )


def dqn_log_metrics_to_dataframe(
        function,
        model,
        predict_func,
        env,
        episode_index,
        games = 1000,
        dataframe = None
):
    if dataframe is None:
        dataframe = pd.DataFrame(
            {
                'training_episode_index': pd.Series(dtype='int'),
                'mean_score': pd.Series(dtype='float'),
                'mean_time_per_episode': pd.Series(dtype='float'),
                'win_rate': pd.Series(dtype='float'),
                'mean_steps_per_episode': pd.Series(dtype='float'),
                'mean_time_per_step': pd.Series(dtype='float')
            }
        )

    (
        mean_score,
        mean_time_per_episode,
        mean_steps_per_episode,
        mean_time_per_step,
        win_rate
     ) = function(env, model, predict_func, games)

    print(f"Episode {episode_index}: Mean Score: {mean_score}, Mean Time: {mean_time_per_episode}, Win Rate: {win_rate}, Mean Steps: {mean_steps_per_episode}, Mean Time per Step: {mean_time_per_step}")

    dataframe = pd.concat([
        dataframe,
        pd.DataFrame([{
            'training_episode_index': episode_index,
            'mean_score': mean_score,
            'mean_time_per_episode': mean_time_per_episode,
            'win_rate': win_rate,
            'mean_steps_per_episode': mean_steps_per_episode,
            'mean_time_per_step': mean_time_per_step
        }])
    ], ignore_index=True)

    return dataframe


def plot_dqn_csv_data(file_path):
    """
    Lit les données d'un fichier CSV et crée des graphiques pour analyser les performances d'entraînement.

    Arguments:
        file_path (str): Le chemin du fichier CSV.
    """
    # Lire le fichier CSV
    data = pd.read_csv(file_path)

    # Définir les colonnes importantes
    x = data['training_episode_index']
    metrics = {
        'Mean Score': data['mean_score'],
        'Mean Time per Episode': data['mean_time_per_episode'],
        'Win Rate': data['win_rate'],
        'Mean Steps per Episode': data['mean_steps_per_episode'],
        'Mean Time per Step': data['mean_time_per_step']
    }

    # Créer des graphiques
    plt.figure(figsize=(15, 10))

    for i, (label, y) in enumerate(metrics.items()):
        plt.subplot(3, 2, i + 1)  # Disposition des sous-graphiques
        plt.plot(x, y, marker='o')
        plt.title(label)
        plt.xlabel('Training Episode Index')
        plt.ylabel(label)
        plt.grid(True)

    plt.tight_layout()
    plt.show()
