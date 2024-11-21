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

def play_game_manual():
    """Fonction pour jouer manuellement contre un adversaire aléatoire."""
    env = None #FarkleDQNEnv(num_players=2, target_score=5000)
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
