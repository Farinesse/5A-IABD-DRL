import numpy as np
import random

class FarkleEnv:
    def __init__(self, num_players=1, target_score=10000):
        self.num_players = num_players
        self.target_score = target_score
        self.reset()

    def reset(self):
        self.scores = [0] * self.num_players
        self.current_player = 0
        self.round_score = 0
        self.remaining_dice = 6
        self.game_over = False
        return self.get_observation()

    def get_observation(self):
        """Retourne l'état actuel sous forme de vecteur."""
        obs = np.array([self.current_player, self.round_score, self.remaining_dice] + self.scores)
        return obs

    def roll_dice(self, num_dice):
        """Lance un nombre donné de dés."""
        return [random.randint(1, 6) for _ in range(num_dice)]

    def calculate_score(self, dice_roll):
        """Calcule les points selon les dés lancés."""
        counts = [dice_roll.count(i) for i in range(1, 7)]
        score = 0
        score += counts[0] * 100  # 1 vaut 100 points
        score += counts[4] * 50   # 5 vaut 50 points

        for i in range(6):
            if counts[i] >= 3:
                score += (i + 1) * 100 if i > 0 else 1000  # Trois dés identiques
        return score

    def step(self, action):
        """Exécute une action, calcule la récompense et passe au tour suivant si nécessaire."""
        dice_roll = self.roll_dice(self.remaining_dice)

        kept_dice = [dice_roll[i] for i in range(len(dice_roll)) if action[i] == 1]
        remaining_dice = [dice_roll[i] for i in range(len(dice_roll)) if action[i] == 0]

        score = self.calculate_score(kept_dice)
        if score == 0:  # Farkle (pas de points)
            self.round_score = 0
            self.next_player()
            return self.get_observation(), -1, True, {}

        self.round_score += score
        self.remaining_dice = len(remaining_dice) if remaining_dice else 6

        done = self.scores[self.current_player] + self.round_score >= self.target_score
        if done:
            self.scores[self.current_player] += self.round_score
            return self.get_observation(), score, True, {}

        return self.get_observation(), score, False, {}

    def next_player(self):
        """Passe au joueur suivant."""
        self.scores[self.current_player] += self.round_score
        self.current_player = (self.current_player + 1) % self.num_players
        self.round_score = 0
        self.remaining_dice = 6

    def render(self):
        """Affiche l'état du jeu."""
        print(f"Joueur {self.current_player + 1}:")
        print(f"Score du tour: {self.round_score}, Dés restants: {self.remaining_dice}")
        print(f"Scores actuels: {self.scores}")

    def is_game_over(self):
        """Vérifie si la partie est terminée."""
        return max(self.scores) >= self.target_score


def epsilon_greedy_action(q_s, available_actions, epsilon):
    """Politique epsilon-greedy pour choisir une action."""
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        return np.argmax(q_s)


def train_agent(env, num_episodes, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1, gamma=0.99, alpha=0.001):
    """Entraîne l'agent à l'aide de la stratégie epsilon-greedy."""
    q_table = {}  # Une table Q simple pour la démonstration

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(env.remaining_dice)  # Initialiser les valeurs Q

            available_actions = [i for i in range(env.remaining_dice)]  # Liste des actions disponibles
            action = epsilon_greedy_action(q_table[state], available_actions, epsilon)

            next_state, reward, done, _ = env.step([1 if i == action else 0 for i in range(env.remaining_dice)])

            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.remaining_dice)

            # Mise à jour de la Q-table
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        env.render()


# Lancer l'entraînement
if __name__ == "__main__":
    env = FarkleEnv(num_players=2)
    train_agent(env, num_episodes=1000)
