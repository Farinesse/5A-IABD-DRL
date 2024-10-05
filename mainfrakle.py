import random

from environment.farkel_env import FarkleDQNEnv


def human_vs_random():
    env = FarkleDQNEnv(num_players=2)  # Jeu à 2 joueurs : 1 humain et 1 agent random
    done = False
    current_player = 0  # 0 = Humain, 1 = Agent Random

    while not env.is_game_over():
        print(f"\n==== Tour du Joueur {current_player + 1} ====")

        # Lancer les dés au début du tour
        env.roll_dice(env.remaining_dice)
        env.render()

        if current_player == 0:  # Tour du joueur humain
            print("\nVotre tour!")
            while True:
                print(f"Dés disponibles : {env.dice_roll}")
                # Demander à l'utilisateur quels dés garder
                kept_dice_input = input(f"Quels dés voulez-vous garder ? (ex: 110000 pour garder les 2 premiers dés) : ")
                if len(kept_dice_input) == len(env.dice_roll):
                    try:
                        kept_dice = [int(c) for c in kept_dice_input]
                        break
                    except ValueError:
                        print("Entrée invalide. Veuillez entrer une séquence de 0 et 1.")
                else:
                    print("Longueur d'entrée invalide. Réessayez.")

            stop = input("Voulez-vous arrêter ce tour ? (oui/non) : ").strip().lower() == "oui"
            action = kept_dice + [1 if stop else 0]  # Combinaison de dés gardés et de l'action stop ou continue

            # Utiliser directement l'action binaire pour le joueur humain
            _, reward, done, info = env.step(action)

        else:  # Tour de l'agent random
            print("\nTour de l'agent Random!")
            available_actions = env.available_actions_ids()
            action_id = random.choice(available_actions)
            action = env.decode_action(action_id)

            # Utiliser l'ID d'action pour l'agent random
            _, reward, done, info = env.step(action_id)

        if 'farkle' in info:
            print("FARKLE! Vous avez perdu les points du tour.")
        elif 'win' in info:
            print(f"Joueur {current_player + 1} a gagné la partie!")

        # Passer au joueur suivant si le joueur a choisi de s'arrêter ou s'il y a un Farkle
        if env.last_action_stop or 'farkle' in info:
            current_player = (current_player + 1) % 2

    # Fin de la partie
    print("\nPartie terminée!")
    env.render()





if __name__ == "__main__":
    human_vs_random()
