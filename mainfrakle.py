import random

from environment.farkle import FarkleEnv


def human_agent(env):
    """Agent humain qui choisit les actions et les dés à garder."""
    while True:
        try:
            action = int(input("Choisissez votre action (0 = S'arrêter et prendre les points, 1 = Continuer à lancer les dés) : "))
            if action in [0, 1]:
                return action
            else:
                print("Action invalide. Veuillez choisir 0 ou 1.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer 0 ou 1.")

def random_agent(env):
    """Agent aléatoire qui choisit de manière aléatoire."""
    return random.choice([0, 1])

def play_farkle(env, agent1, agent2, num_episodes=1):
    """Jouer à Farkle avec deux agents."""
    for episode in range(num_episodes):
        print(f"\n--- Épisode {episode + 1} ---")
        done = False
        env.reset()

        while not env.is_game_over():
            env.render()

            if env.current_player == 0:
                action = agent1(env)  # Tour du premier agent (peut être un humain ou aléatoire)
            else:
                action = agent2(env)  # Tour du second agent (peut être un humain ou aléatoire)

            _, reward, done, _ = env.step(action)
            print(f"Récompense: {reward}")

        print(f"Fin de l'épisode {episode + 1}.\n")

    print("Le jeu est terminé.")
    env.render()


if __name__ == "__main__":
    # Initialiser l'environnement Farkle
    env = FarkleEnv(num_players=2, target_score=10000)

    # Choisir les agents (humain ou aléatoire)
    print("Bienvenue dans le jeu de Farkle !")

    agent1_type = input("Agent 1: Choisissez le type d'agent (1 = Humain, 2 = Aléatoire) : ")
    agent2_type = input("Agent 2: Choisissez le type d'agent (1 = Humain, 2 = Aléatoire) : ")

    agent1 = human_agent if agent1_type == '1' else random_agent
    agent2 = human_agent if agent2_type == '1' else random_agent

    # Jouer le jeu
    play_farkle(env, agent1, agent2, num_episodes=1)
