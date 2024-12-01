import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from statistics import mean

from colorama import Fore
from tqdm import tqdm

from GUI.test import predict_func, epsilon_greedy_action_bis


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


def epsilon_greedy_action(q_s: tf.Tensor,
                          mask: tf.Tensor,
                          available_actions: np.ndarray,
                          epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        # Masquer les actions invalides avec -inf
        inverted_mask = tf.constant(1.0) - mask
        masked_q_s = q_s * mask + (-1e9) * inverted_mask

        # S'assurer de choisir parmi les actions valides
        masked_q_values = masked_q_s.numpy()
        valid_q_values = masked_q_values[available_actions]
        return available_actions[np.argmax(valid_q_values)]

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


def play(game, player_x, print_game=True):
    """
    Permet à deux joueurs (humain ou IA) de jouer une partie de TicTacToe.
    """
    if print_game:
        game.display()

    letter = 'X'  # Le premier joueur commence avec 'X'
    while not game.is_game_over():

        square = player_x(game)
        game.step(square)  # Exécute l'action
        print("state : ", game.state_description())
        print("ACTION : ", game.available_actions_ids())

        print("ACTION MASK : ", game.action_mask())

        if print_game:
            game.display()

        if game.is_game_over():
            if print_game:
                if game.score() == 1.0:
                    print(f'{letter} a gagné!')
                elif game.score() == 0.0:
                    print('C\'est un match nul!')
                else :
                    print(f'O a gagné!')
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


def play_line_grid_world(game, player_human=None, player_random=None, print_game=True):
    """
    Permet à un humain et un agent aléatoire de jouer à LineWorld.
    """
    if print_game:
        game.display()

    while not game.is_game_over():
        if player_human:
            action = player_human(game)
            game.step(action)
            if print_game:
                game.display()

            if game.is_game_over():
                if print_game:
                    print("Jeu terminé ! L'agent est dans un état terminal.")
                break

        if player_random:
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
        while not env.is_game_over() and nb_turns < 100:
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_s = predict_func(model, s_tensor)


            a = epsilon_greedy_action(q_s.numpy(), mask_tensor, env.available_actions_ids(), 0.000001)
            if a not in env.available_actions_ids():
                print(f"Action {a} invalide, prise aléatoire à la place.")
                a = np.random.choice(env.available_actions_ids())

            env.step(a)
            nb_turns += 1

        end_time = time.time()
        if nb_turns == 100:
            episode_scores.append(-1)
        else:
            episode_scores.append(env.score(testing=False))
        episode_time = end_time - start_time
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


def log_metrics_to_dataframe(
        function,
        model,
        predict_func,
        env,
        episode_index,
        games = 100,
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

def plot_csv_data(file_path):
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


def play_with_mcts(env, agent, episodes=100):
    """Fonction pour jouer plusieurs épisodes et collecter les statistiques."""
    episode_scores = []
    episode_times = []
    episode_steps = []
    step_times = []
    total_time = 0

    for episode in range(episodes):
        env.reset()
        nb_turns = 0

        start_time = time.time()
        while not env.is_game_over() and nb_turns < 100:
            # Gestion des tours pour Farkle

            action = agent.select_action(env)
            env.step(action)
            env.display()
            nb_turns += 1

        end_time = time.time()

        # Enregistrement des statistiques
        if nb_turns == 100:  # Limite de tours atteinte
            episode_scores.append(-1)
        else:
            episode_scores.append(env.score())

        episode_time = end_time - start_time
        episode_times.append(episode_time)
        total_time += episode_time
        episode_steps.append(nb_turns)
        step_times.append(episode_time / nb_turns if nb_turns > 0 else 0)

    return (
        mean(episode_scores),
        mean(episode_times),
        mean(episode_steps),
        mean(step_times),
        episode_scores.count(1.0) / episodes
    )


def log_metrics_to_dataframe_mcts(function, agent, env, episode_index, games=100, dataframe=None):
    """Enregistre les métriques dans un DataFrame."""
    if dataframe is None:
        dataframe = pd.DataFrame({
            'training_episode_index': pd.Series(dtype='int'),
            'mean_score': pd.Series(dtype='float'),
            'mean_time_per_episode': pd.Series(dtype='float'),
            'win_rate': pd.Series(dtype='float'),
            'mean_steps_per_episode': pd.Series(dtype='float'),
            'mean_time_per_step': pd.Series(dtype='float')
        })

    (
        mean_score,
        mean_time_per_episode,
        mean_steps_per_episode,
        mean_time_per_step,
        win_rate
    ) = function(env, agent, games)

    print(f"Episode {episode_index}: Mean Score: {mean_score:.3f}, Mean Time: {mean_time_per_episode:.3f}, "
          f"Win Rate: {win_rate:.2%}, Mean Steps: {mean_steps_per_episode:.1f}, "
          f"Mean Time per Step: {mean_time_per_step:.3f}")

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


def plot_mcts_metrics(file_path, title="MCTS Training Metrics", agent=None, custom_dict=None):
    """Plot les métriques d'entraînement."""
    data = pd.read_csv(file_path)

    x = data['training_episode_index']
    metrics = {
        'Mean Score': data['mean_score'],
        'Mean Time per Episode': data['mean_time_per_episode'],
        'Win Rate': data['win_rate'],
        'Mean Steps per Episode': data['mean_steps_per_episode'],
        'Mean Time per Step': data['mean_time_per_step']
    }

    # Création du plot avec matplotlib
    plt.figure(figsize=(15, 10))
    for i, (metric_name, metric_data) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(x, metric_data)
        plt.title(metric_name)
        plt.xlabel('Training Episode')
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()


def play_agent_vs_random_tictactoe(env, model, num_games: int = 100) -> None:
    try:
        wins = 0
        losses = 0
        draws = 0

        print(Fore.CYAN + f"\n=== Agent vs Random sur {num_games} parties ===")

        for game in range(num_games):
            env.reset()
            game_done = False

            while not game_done:
                # Tour de l'agent
                state = env.state_description()
                s_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                mask = env.action_mask()

                q_s = predict_func(model, s_tensor)
                print(q_s,mask,s_tensor)

                masked_q_values = q_s[0].numpy() * mask - 1e9 * (1 - mask)
                action = np.argmax(masked_q_values)

                reward = env.step(action)
                env.display()
                game_done = env.is_game_over()

                if game_done:
                    break

            # Résultat de la partie
            if env.score() > 0:
                wins += 1
            elif env.score() < 0:
                losses += 1
            else:
                draws += 1

            if (game + 1) % 10 == 0:
                print(Fore.GREEN + f"\rParties jouées : {game + 1}/{num_games}", end="")

        print(Fore.GREEN + "\n\nRésultats :")
        print(f"Victoires : {wins} ({wins / num_games * 100:.1f}%)")
        print(f"Défaites : {losses} ({losses / num_games * 100:.1f}%)")
        print(f"Nuls : {draws} ({draws / num_games * 100:.1f}%)")

    except Exception as e:
        print(Fore.RED + f"\nErreur lors du chargement du modèle : {e}")
        print(Fore.YELLOW + "Le modèle n'a pas pu être chargé")


def play_with_agent_gridworld(env, model, num_games: int = 100) -> None:
    try:
        wins = 0
        losses = 0
        draws = 0

        print(Fore.CYAN + f"\n=== Agent vs Random sur {num_games} parties ===")

        for game in range(num_games):
            env.reset()
            game_done = False
            cumulative_reward = 0

            while not game_done:
                state = env.state_description()
                s_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                mask = env.action_mask()
                mask_array = np.array(mask, dtype=np.float32)

                q_s = predict_func(model, s_tensor)
                print(q_s, mask, s_tensor)

                masked_q_values = q_s[0].numpy() * mask_array - 1e9 * (1 - mask_array)
                action = np.argmax(masked_q_values)

                reward = env.step(action)
                env.display()
                game_done = env.is_game_over()

                # Gestion de la récompense qui peut être None
                if reward is not None:
                    cumulative_reward += reward

            # Évaluation de la performance à la fin de la partie
            if cumulative_reward > 0:
                wins += 1
            elif cumulative_reward < 0:
                losses += 1
            else:
                draws += 1

            if (game + 1) % 10 == 0:
                print(Fore.GREEN + f"\rParties jouées : {game + 1}/{num_games}", end="")

        print(Fore.GREEN + "\n\nRésultats :")
        print(f"Victoires : {wins} ({wins / num_games * 100:.1f}%)")
        print(f"Défaites : {losses} ({losses / num_games * 100:.1f}%)")
        print(f"Nuls : {draws} ({draws / num_games * 100:.1f}%)")

    except Exception as e:
        print(Fore.RED + f"\nErreur lors du chargement du modèle : {e}")
        print(Fore.YELLOW + "Le modèle n'a pas pu être chargé")


def play_with_agent_lineworld(env, model, num_games: int = 100) -> None:
    try:
        wins = 0
        losses = 0
        draws = 0

        print(f"\n=== Agent vs Random sur {num_games} parties ===")

        for game in range(num_games):
            env.reset()
            game_done = False
            cumulative_reward = 0

            while not game_done:
                state = env.state_description()
                s_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                env.display()
                mask = env.action_mask()
                print(f"Mask: {mask}")
                print(f"State tensor: {s_tensor}")

                try:
                    q_s = predict_func(model, s_tensor)
                    print(f"Q-values: {q_s}")
                    masked_q_values = q_s[0].numpy() * mask - 1e9 * (1 - mask)
                    action = np.argmax(masked_q_values)
                    reward = env.step(action)
                    game_done = env.is_game_over()

                    # Accumulation des récompenses non-nulles
                    if reward is not None:
                        cumulative_reward += reward
                except Exception as e:
                    print(f"Erreur lors de la prédiction du modèle : {e}")
                    break

            # Évaluation de la performance
            if cumulative_reward > 0:
                wins += 1
            elif cumulative_reward < 0:
                losses += 1
            else:
                draws += 1

            # Affichage de la progression
            if (game + 1) % 10 == 0:
                print(f"\rParties jouées : {game + 1}/{num_games} (Victoires: {wins}, Pertes: {losses}, Nuls: {draws})",
                      end="")

        # Statistiques finales
        print("\n\nRésultats finaux :")
        print(f"Victoires : {wins} ({wins / num_games * 100:.1f}%)")
        print(f"Défaites : {losses} ({losses / num_games * 100:.1f}%)")
        print(f"Nuls : {draws} ({draws / num_games * 100:.1f}%)")
        print(f"Score moyen par partie : {cumulative_reward / num_games:.2f}")

    except Exception as e:
        print(f"\nErreur lors du chargement du modèle : {e}")
        print("Le modèle n'a pas pu être chargé")
