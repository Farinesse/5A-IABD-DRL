import numpy as np

import random
from QLearning.qlearning import tabular_q_learning
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import numpy as np

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


def play_with_q_agent(env_type, random_agent_fn, *env_args, if2X2=False):
    """
    Joue un match avec un agent Q-Learning contre un agent aléatoire.
    :param env_type: Le type de l'environnement (doit respecter GameEnv)
    :param random_agent_fn: Fonction qui définit l'agent aléatoire
    :param env_args: Arguments pour initialiser l'environnement
    """
    # Entraînement de l'agent Q-Learning
    print("Entraînement de l'agent Q-Learning...")
    Pi, Q = tabular_q_learning(lambda: env_type(*env_args), alpha=0.1, epsilon=0.1, gamma=0.999, nb_iter=10000)

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

        env.display()
        if if2X2 :
            if not env.is_game_over():
                # L'agent aléatoire joue
                random_action = random_agent_fn(env)
                env.step(random_action)
                env.display()

          # Affiche l'état du jeu après chaque action

    print("Jeu terminé. Score final:", env.score())

def play_with_dqn(env, model, random_agent=None, episodes=1):
    total_rewards = 0  # Initialiser le total des récompenses pour calculer le score moyen

    for episode in range(episodes):
        env.reset()
        done = False
        total_reward = 0

        while not env.is_game_over():
            state = env.state_description()  # Obtenez l'état actuel
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)  # Transformez l'état en tenseur

            # L'agent DQN choisit une action
            q_values = model(state_tensor)
            action = tf.argmax(q_values[0]).numpy()  # Choisit l'action avec la plus haute valeur Q

            if action not in env.available_actions_ids():
                print(f"Action {action} invalide, prise aléatoire.")
                if random_agent:
                    action = random_agent(env)  # Utilisez un agent aléatoire si l'action est invalide
                else:
                    action = np.random.choice(
                        env.available_actions_ids())  # Si pas de random_agent, choisissez aléatoirement

            # Faites l'action dans l'environnement
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            total_reward += reward

            # Afficher l'état du jeu après chaque mouvement
            env.display()

        print(f"Épisode {episode + 1}/{episodes} terminé, Total Reward: {total_reward}")
        total_rewards += total_reward  # Ajouter la récompense de cet épisode au total des récompenses

    # Calculer et afficher le score moyen après tous les épisodes
    mean_score = total_rewards / episodes
    print(f"Score moyen après {episodes} épisodes : {mean_score}")

'''
def play_dqn_vs_random(env, dqn_model, random_agent_func, episodes):
    """
    Cette fonction permet de jouer un agent DQN contre un agent random sur plusieurs épisodes.
    """
    dqn_wins = 0
    random_wins = 0
    draws = 0
    total_dqn_score = 0
    invalid_actions = 0  # Compteur d'actions invalides

    for episode in range(episodes):
        env.reset()
        dqn_turn = True

        while not env.is_game_over():
            state = env.state_description()
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

            if dqn_turn:
                q_values = dqn_model(state_tensor)
                action = tf.argmax(q_values[0]).numpy()

                if action not in env.available_actions_ids():
                    print(f"Action {action} invalide, prise aléatoire par l'agent DQN.")
                    invalid_actions += 1
                    action = random_agent_func(env)
                print(f"Agent DQN choisit l'action {action}")
            else:
                action = random_agent_func(env)
                print(f"Agent Random choisit l'action {action}")

            prev_score = env.score()
            env.step(action)
            total_reward = env.score() - prev_score
            total_dqn_score += total_reward if dqn_turn else 0  # Ajouter au score DQN si c'était son tour
            env.display()
            dqn_turn = not dqn_turn

        # Résultat de la partie
        if env.score() > 0:
            dqn_wins += 1
            print(f"L'agent DQN gagne l'épisode {episode + 1}.")
        elif env.score() < 0:
            random_wins += 1
            print(f"L'agent Random gagne l'épisode {episode + 1}.")
        else:
            draws += 1
            print(f"Match nul dans l'épisode {episode + 1}.")

    # Résumé après tous les épisodes
    print(f"\nAprès {episodes} épisodes :")
    print(f"Agent DQN a gagné {dqn_wins} fois.")
    print(f"Agent Random a gagné {random_wins} fois.")
    print(f"Il y a eu {draws} match(s) nul(s).")
    print(f"Nombre total d'actions invalides prises par l'agent DQN : {invalid_actions}")
    print(f"Score total de l'agent DQN : {total_dqn_score}")'''
def play_dqn_vs_random(env, dqn_model, random_agent_func, episodes):
    """
    Cette fonction permet de faire jouer un agent DQN contre un agent random sur plusieurs épisodes.
    L'agent random joue automatiquement après l'agent DQN via la méthode 'step()' de l'environnement.
    """
    dqn_wins = 0
    random_wins = 0
    draws = 0
    total_dqn_score = 0
    invalid_actions = 0  # Compteur d'actions invalides

    for episode in range(episodes):
        env.reset()

        while not env.is_game_over():
            state = env.state_description()
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

            # Tour de l'agent DQN
            q_values = dqn_model(state_tensor)
            action = tf.argmax(q_values[0]).numpy()

            # Vérifier si l'action est valide
            if action not in env.available_actions_ids():
                print(f"Action {action} invalide, prise aléatoire par l'agent DQN.")
                invalid_actions += 1
                action = random_agent_func(env)  # Action aléatoire en cas d'invalidité

            print(f"Agent DQN choisit l'action {action}")
            env.step(action)  # L'agent random joue automatiquement après le DQN dans 'step()'

            total_reward = env.score()
            total_dqn_score += total_reward  # Ajouter au score DQN uniquement si c'était son tour
            env.display()

        # Résultat de la partie
        if env.score() > 0:
            dqn_wins += 1
            print(f"L'agent DQN gagne l'épisode {episode + 1}.")
        elif env.score() < 0:
            random_wins += 1
            print(f"L'agent Random gagne l'épisode {episode + 1}.")
        else:
            draws += 1
            print(f"Match nul dans l'épisode {episode + 1}.")

    # Résumé après tous les épisodes
    print(f"\nAprès {episodes} épisodes :")
    print(f"Agent DQN a gagné {dqn_wins} fois.")
    print(f"Agent Random a gagné {random_wins} fois.")
    print(f"Il y a eu {draws} match(s) nul(s).")
    print(f"Nombre total d'actions invalides prises par l'agent DQN : {invalid_actions}")
    print(f"Score total de l'agent DQN : {total_dqn_score}")


def farkel_random_player(env):
    """
    Crée un joueur qui joue de manière aléatoire au jeu Farkel.

    :param env: L'environnement FarkleEnv
    :return: Une action aléatoire valide sous forme de liste de 7 bits
    """
    valid_actions = env.get_valid_actions()
    valid_indices = np.where(valid_actions == 1)[0]

    if len(valid_indices) == 0:
        print("Aucune action valide disponible, retour à l'action par défaut.")
        return [0, 0, 0, 0, 0, 0, 0]  # Action par défaut si aucune action valide n'est disponible

    # Choisir un index d'action aléatoire parmi les actions valides
    random_action_index = random.choice(valid_indices)

    # Convertir l'index d'action en une liste de 7 bits
    random_action = [int(b) for b in format(random_action_index, '07b')]

    return random_action


def play_farkel_human_vs_qlearning(agent, env, gui):

    gui.env = env

    while not env.game_over:
        # Tour du joueur humain
        print("Tour du joueur humain")
        gui.update_display()
        gui.master.update()
        action = gui.wait_for_action()
        state, reward, done, _, _ = env.step(action)
        print(f"Récompense du joueur humain: {reward}")

        if done:
            break

        # Tour de l'agent Q-learning
        print("Tour de l'agent Q-learning")
        action = agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        print(f"Récompense de l'agent Q-learning: {reward}")
        gui.update_display()
        gui.master.update()

    print("Partie terminée")
    print(f"Score final - Joueur: {env.scores[0]}, Agent Q-learning: {env.scores[1]}")
    gui.master.mainloop()


def play_farkel_human_vs_random(env, gui, root):
    while not env.game_over:
        # Tour du joueur humain
        print("Tour du joueur humain")
        gui.update_display()
        root.update()
        action = gui.wait_for_action()
        observation, reward, done, _, info = env.step(action)
        print(f"Récompense du joueur humain: {reward}")
        gui.update_display()
        root.update()

        if done:
            break

        # Tour de l'agent aléatoire
        print("Tour de l'agent aléatoire")
        random_action = farkel_random_player(env)
        observation, reward, done, _, info = env.step(random_action)
        print(f"Récompense de l'agent aléatoire: {reward}")
        gui.update_display()
        root.update()

        if done:
            break

    print("Partie terminée")
    print(f"Score final - Joueur: {env.scores[0]}, Agent aléatoire: {env.scores[1]}")