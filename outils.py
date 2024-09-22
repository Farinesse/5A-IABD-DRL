'''def human_move(game):
    """
    Permet à l'humain de choisir une case sur le plateau (de 0 à 8).
    """
    valid_square = False
    val = None
    while not valid_square:
        square = input('Entrez un choix de case (0-8) : ')
        try:
            val = int(square)
            if val not in game.available_actions():
                raise ValueError
            valid_square = True
        except ValueError:
            print("Case invalide, essayez à nouveau.")
    return val'''

'''def play(game, player_x, player_o, print_game=True):
    """
    Permet à deux joueurs (humain ou IA) de jouer une partie de TicTacToe.
    """
    if print_game:
        game.display()

    letter = 'X'  # Le premier joueur commence avec 'X'
    while not game.is_game_over():
        if letter == 'O':
            # Utilisation de l'agent random pour O
            square = player_o(game)
        else:
            # Utilisation du joueur humain pour X
            square = player_x(game)

        state, reward, done = game.step(square, letter)
        if print_game:
            print(f'{letter} a joué sur la case {square}')
            game.display()
            print('')

        if done:
            if print_game:
                if reward == 1.0:
                    print(f'{letter} a gagné!')
                else:
                    print('C\'est un match nul!')
            return letter  # Retourne le gagnant ou None en cas de match nul

        letter = 'O' if letter == 'X' else 'X'  # Change de joueur'''

'''
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
        action = int(input("Votre choix (0, 1, 2) : "))
        if action in game.available_actions():
            valid_action = True
        else:
            print("Action invalide ! Veuillez choisir parmi les actions disponibles.")
    return action


def play_line_world(game, player_human, player_random, print_game=True):
    """
    Permet à un humain et un agent aléatoire de jouer à LineWorld.
    """
    if print_game:
        game.display()

    while not game.is_game_over():
        action = player_human(game)  # Joueur humain
        next_state, reward, done = game.step(action)
        if print_game:
            print(f"État suivant: {next_state}, Récompense: {reward}, Terminé: {done}")
            game.display()

        if done:
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
        action = int(input("Votre choix (0, 1, 2, 3) : "))
        if action in game.available_actions():
            valid_action = True
        else:
            print("Action invalide ! Veuillez choisir parmi les actions disponibles.")
    return action


def play_grid_world(game, player_human, player_random, print_game=True):
    """
    Permet à un humain et un agent aléatoire de jouer à GridWorld.
    """
    if print_game:
        game.display()

    while not game.is_game_over():
        action = player_human(game)  # Joueur humain
        next_state, reward, done = game.step(action)
        if print_game:
            print(f"État suivant: {next_state}, Récompense: {reward}, Terminé: {done}")
            game.display()

        if done:
            if print_game:
                print("Jeu terminé ! L'agent est dans un état terminal.")
            break
'''

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

    letter = 'X'  # Le premier joueur commence avec 'X'
    while not game.is_game_over():
        if letter == 'O':
            # Utilisation de l'agent random pour O
            square = player_o(game)
        else:
            # Utilisation du joueur humain pour X
            square = player_x(game)

        # Exécute l'action sans décompacter
        game.step(square)  # On n'attend plus de retour ici

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

        letter = 'O' if letter == 'X' else 'X'  # Change de joueur


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
        action = player_human(game)  # Joueur humain
        game.step(action)  # Effectue l'action du joueur humain
        if print_game:
            game.display()

        if game.is_game_over():  # Si le jeu est terminé
            if print_game:
                print("Jeu terminé ! L'agent est dans un état terminal.")
            break

        # Le joueur aléatoire effectue une action
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
        action = player_human(game)  # Joueur humain
        game.step(action)  # Effectue l'action du joueur humain
        if print_game:
            game.display()

        if game.is_game_over():  # Si le jeu est terminé
            if print_game:
                print("Jeu terminé ! L'agent est dans un état terminal.")
            break

        # Le joueur aléatoire effectue une action
        action = player_random(game)
        game.step(action)
        if print_game:
            game.display()

        if game.is_game_over():
            if print_game:
                print("Jeu terminé ! L'agent est dans un état terminal.")
            break

