import tensorflow as tf
from environment.tictactoe import TicTacToe_new
import numpy as np
import os


def saved_tictactoe_model(model_path, num_games=10000):
    # Vérifier si le fichier existe
    if not os.path.exists(model_path):
        print(f"Erreur : Le fichier '{model_path}' n'existe pas.")
        print(f"Répertoire de travail actuel : {os.getcwd()}")
        print("Contenu du répertoire :")
        print(os.listdir())
        return

    try:
        # Charger le modèle sauvegardé
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    # Créer l'environnement
    env = TicTacToe_new()

    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            state = env.state_description()
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

            # Tour de l'agent DQN
            q_values = model(state_tensor)[0].numpy()

            # Appliquer le masque d'action
            action_mask = env.action_mask()
            masked_q_values = q_values * action_mask - 1e9 * (1 - action_mask)

            action = np.argmax(masked_q_values)



            print(f"Agent DQN choisit l'action {action}")

            env.step(action)

            env.display()
            done = env.is_game_over()

        final_score = env.score()
        if final_score == 1.0:
            wins += 1
        elif final_score == -1.0:
            losses += 1
        else:
            draws += 1

    win_rate = wins / num_games
    loss_rate = losses / num_games
    draw_rate = draws / num_games

    print(f"Résultats sur {num_games} parties:")
    print(f"Victoires: {wins} ({win_rate:.2%})")
    print(f"Défaites: {losses} ({loss_rate:.2%})")
    print(f"Matchs nuls: {draws} ({draw_rate:.2%})")

    return win_rate, loss_rate, draw_rate


if __name__ == "__main__":
    # Utiliser un chemin absolu avec un raw string
    model_path = r"C:\Users\farin\PycharmProjects\5A-IABD-DRL\double_dqn_tictactoe_final.h5"

    # Ou vous pouvez utiliser des doubles backslashes
    # model_path = "C:\\Users\\farin\\PycharmProjects\\5A-IABD-DRL\\dqn_model_tictactoe_t1.h5"

    saved_tictactoe_model(model_path)