
import math

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
        print("state : ",game.state_description())
        print("ACTION : " , game.available_actions_ids())

        print("ACTION MASK : ",game.action_mask())


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
