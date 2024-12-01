import time
from environment.FarkelEnv import FarkleDQNEnv
from utils.animations import rolling_dice_animation
from utils.scores import save_game_history
from utils.styles import print_colored


def farkle_menu():
    """Menu des modes de jeu pour Farkle."""
    while True:
        print_colored("\n=== MODE DE JEU ===", "cyan")
        print_colored("1. Humain vs Random", "yellow")
        print_colored("2. Random vs Random", "yellow")
        print_colored("3. Humain vs Agent (algorithme)", "yellow")
        print_colored("4. Agent (algorithme) vs Random", "yellow")
        print_colored("5. Retour", "red")

        choice = input("Choisissez un mode de jeu : ")
        if choice == "1":
            play_human_vs_random()
        elif choice == "2":
            play_random_vs_random()
        elif choice == "3":
            play_human_vs_agent()
        elif choice == "4":
            play_agent_vs_random()
        elif choice == "5":
            print_colored("Retour au menu principal.", "green")
            break
        else:
            print_colored("Option invalide. Réessayez.", "red")


def play_human_vs_random():
    """Mode Humain vs Random."""
    print_colored("\n=== HUMAIN VS RANDOM ===", "cyan")
    env = FarkleDQNEnv(target_score=5000)
    history = []

    while not env.is_game_over():
        print_colored("\n🎮 Joueur 1 : Votre tour !", "yellow")
        env.display()
        action = input("Entrez une action valide (0 pour arrêter) : ")
        try:
            action = int(action)
            valid_actions = env.available_actions_ids()
            if action not in valid_actions:
                raise ValueError("Action invalide.")
            _, reward, done, _, info = env.step(action)
            if info.get("farkle"):
                print_colored(f"💥 Farkle ! Vous avez perdu {info['lost_points']} points.", "red")
            elif info.get("stopped"):
                print_colored(f"🛑 Tour terminé. Points gagnés : {reward}.", "green")
            elif info.get("win"):
                print_colored("🎉 Félicitations ! Vous avez gagné.", "cyan")
                break
        except ValueError as e:
            print_colored(f"Erreur : {str(e)}", "red")
            continue

        print_colored("\n🎮 Joueur 2 : Random joue...", "yellow")
        env.play_random_turn()
        env.display()

    history.append((env.scores[0], env.scores[1]))
    save_game_history(history)
    print_colored("\nPartie terminée. Scores sauvegardés.", "green")


def play_random_vs_random():
    """Mode Random vs Random."""
    print_colored("\n=== RANDOM VS RANDOM ===", "cyan")
    env = FarkleDQNEnv(target_score=5000)

    while not env.is_game_over():
        print_colored("\n🎲 Random 1 joue...", "yellow")
        action1 = env.get_random_action()
        env.step(action1)
        env.display()
        if env.is_game_over():
            break

        print_colored("\n🎲 Random 2 joue...", "yellow")
        action2 = env.get_random_action()
        env.step(action2)
        env.display()

    print_colored("\n=== Scores finaux ===", "cyan")
    print_colored(f"Joueur 1 (Random) : {env.scores[0]}", "green")
    print_colored(f"Joueur 2 (Random) : {env.scores[1]}", "green")


def play_human_vs_agent():
    """Mode Humain vs Agent (algorithme)."""
    print_colored("\n=== HUMAIN VS AGENT ===", "cyan")
    model_path = input("Entrez le chemin du modèle pré-entraîné : ")
    print_colored(f"Chargement du modèle depuis {model_path}...", "yellow")

    # Simule le chargement du modèle - ajouter la logique réelle ici
    # model = keras.models.load_model(model_path)

    env = FarkleDQNEnv(target_score=5000)
    print_colored("Modèle chargé. Début de la partie.", "green")

    while not env.is_game_over():
        print_colored("\n🎮 Votre tour !", "yellow")
        env.display()
        action = input("Entrez une action valide (0 pour arrêter) : ")
        try:
            action = int(action)
            valid_actions = env.available_actions_ids()
            if action not in valid_actions:
                raise ValueError("Action invalide.")
            _, reward, done, _, info = env.step(action)
            if info.get("farkle"):
                print_colored(f"💥 Farkle ! Vous avez perdu {info['lost_points']} points.", "red")
            elif info.get("stopped"):
                print_colored(f"🛑 Tour terminé. Points gagnés : {reward}.", "green")
            elif info.get("win"):
                print_colored("🎉 Félicitations ! Vous avez gagné.", "cyan")
                break
        except ValueError as e:
            print_colored(f"Erreur : {str(e)}", "red")
            continue

        print_colored("\n🎮 L'agent joue...", "yellow")
        action_agent = env.get_random_action()  # Remplacez par `model.predict(env.state_description())`
        env.step(action_agent)
        env.display()


def play_agent_vs_random():
    """Mode Agent (algorithme) vs Random."""
    print_colored("\n=== AGENT VS RANDOM ===", "cyan")
    model_path = input("Entrez le chemin du modèle pré-entraîné : ")
    print_colored(f"Chargement du modèle depuis {model_path}...", "yellow")

    # Simule le chargement du modèle - ajouter la logique réelle ici
    # model = keras.models.load_model(model_path)

    env = FarkleDQNEnv(target_score=5000)
    print_colored("Modèle chargé. Début de la partie.", "green")

    while not env.is_game_over():
        print_colored("\n🎮 L'agent joue...", "yellow")
        action_agent = env.get_random_action()  # Remplacez par `model.predict(env.state_description())`
        env.step(action_agent)
        env.display()
        if env.is_game_over():
            break

        print_colored("\n🎮 Random joue...", "yellow")
        action_random = env.get_random_action()
        env.step(action_random)
        env.display()
