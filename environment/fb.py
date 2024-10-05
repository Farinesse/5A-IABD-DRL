import random
import numpy as np


class FarkleEnv:
    def __init__(self, num_players=1, target_score=10000):
        self.num_players = num_players
        self.target_score = target_score
        self.reset()

    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode."""
        self.scores = [0] * self.num_players
        self.current_player = 0
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = []
        self.game_over = False
        self.last_action_stop = False
        return self.get_observation()

    def get_observation(self):
        """Retourne une observation de l'état actuel sous forme de vecteur."""
        # Assurons-nous que dice_roll a toujours 6 éléments
        padded_dice = self.dice_roll + [0] * (6 - len(self.dice_roll))
        obs = np.array([
                           self.current_player,
                           self.round_score,
                           self.remaining_dice
                       ] + self.scores + padded_dice + [int(self.last_action_stop)])
        return obs

    def roll_dice(self, num_dice):
        """Lance un nombre donné de dés."""
        self.dice_roll = [random.randint(1, 6) for _ in range(num_dice)]
        return self.dice_roll

    def calculate_score(self, dice_roll, use_restriction=True):
        """Calcule les points selon les dés lancés."""
        if not dice_roll:
            return 0

        counts = [dice_roll.count(i) for i in range(1, 7)]
        score = 0

        # Vérification de la suite
        if sorted(dice_roll) == [1, 2, 3, 4, 5, 6]:
            return 1500

        # Vérification des trois paires
        if len(dice_roll) == 6 and counts.count(2) == 3:
            return 1000

        # Calcul des scores pour les groupes de 3 ou plus
        for value, count in enumerate(counts, start=1):
            if count >= 3:
                if value == 1:
                    score += 1000 * (2 ** (count - 3))
                else:
                    score += value * 100 * (2 ** (count - 3))

        # Ajout des points pour les 1 et 5 individuels
        ones_count = counts[0] % 3  # Nombre de 1 restants après les triplets
        fives_count = counts[4] % 3  # Nombre de 5 restants après les triplets

        score += ones_count * 100
        score += fives_count * 50

        if use_restriction:
            # Vérification des dés inutilisables
            for i, count in enumerate(counts):
                if i not in [0, 4] and count > 0 and count < 3:
                    return 0

        return score

    def step(self, action):
        """Exécute une action dans l'environnement."""
        if len(action) != 7:
            raise ValueError("Action invalide. Vous devez entrer une séquence binaire de 7 chiffres.")

        initial_score = self.calculate_score(self.dice_roll, False)

        # Bonus de 500 points pour un Farkle avec tous les dés
        if initial_score == 0 and self.remaining_dice == 6:
            self.round_score += 500  # Ajout du bonus de 500 points
            self.next_player()
            return self.get_observation(), 500, False, {"farkle": True}

        # Sélection des dés gardés
        kept_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action[i] == 1]
        new_score = self.calculate_score(kept_dice)

        # Mise à jour du nombre de dés restants
        self.remaining_dice -= sum(action[:len(self.dice_roll)])

        # Farkle normal (pas de points avec moins de 6 dés)
        if new_score == 0:
            self.round_score = 0
            self.next_player()
            return self.get_observation(), -1, False, {"farkle": True}

        self.round_score += new_score

        # Vérification de la victoire
        if self.scores[self.current_player] + self.round_score >= self.target_score:
            self.scores[self.current_player] += self.round_score
            self.game_over = True
            return self.get_observation(), new_score, True, {"win": True}

        # Gestion de l'action "stop"
        self.last_action_stop = bool(action[-1])
        if self.last_action_stop:
            self.next_player()

        return self.get_observation(), new_score, False, {}

    def next_player(self):
        """Passe au joueur suivant."""
        self.scores[self.current_player] += self.round_score
        self.current_player = (self.current_player + 1) % self.num_players
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = []
        self.last_action_stop = False

    def render(self):
        """Affiche l'état du jeu."""
        print(f"Joueur {self.current_player + 1}:")
        print(f"Dés lancés: {self.dice_roll}")
        print(f"Score du tour: {self.round_score}")
        print(f"Dés restants: {self.remaining_dice}")
        print(f"Scores actuels: {self.scores}")

    def is_game_over(self):
        """Vérifie si la partie est terminée."""
        return self.game_over


# Les fonctions d'agents restent inchangées

def main():
    env = FarkleEnv(num_players=2)
    env.reset()

    while not env.is_game_over():
        print(f"\nTour du joueur {env.current_player + 1}...")

        while True:
            env.dice_roll = env.roll_dice(env.remaining_dice)
            env.render()

            initial_score = env.calculate_score(env.dice_roll, False)

            if initial_score == 0 and env.remaining_dice < 6:
                print("Farkle! Passage au joueur suivant.")
                env.round_score = 0
                env.next_player()
                break

            if env.current_player == 0:
                action = human_agent(env, env.dice_roll)
            else:
                action = random_agent(env, env.dice_roll)

            observation, reward, done, info = env.step(action)
            print(f"Points gagnés ce lancer : {reward}")

            if done:
                print(f"Le joueur {env.current_player + 1} a gagné!")
                break

            if env.last_action_stop or env.remaining_dice == 0:
                break

        if done:
            break

    print(f"\nScores finaux: {env.scores}")
    print(f"Le joueur {env.current_player + 1} a gagné!")

def random_agent(env, dice_roll):
    """Agent aléatoire qui choisit une action binaire aléatoire."""
    return [random.choice([0, 1]) for _ in range(len(dice_roll) + 1)]


def human_agent(env, dice_roll):
    """Agent humain pour choisir les dés à garder ou relancer."""
    print(f"Résultat des dés actuels : {dice_roll}")
    while True:
        try:
            action_str = input(
                f"Entrez une action binaire de {len(dice_roll)} chiffres (0 = relancer, 1 = garder) et 1 pour arrêter le tour : ")
            if len(action_str) == 7 and all(c in '01' for c in action_str):
                return [int(x) for x in action_str]
            else:
                print(f"Action invalide. Veuillez entrer une séquence binaire de {len(dice_roll) + 1} chiffres.")
        except ValueError:
            print(f"Entrée invalide. Veuillez entrer une séquence binaire de {len(dice_roll) + 1} chiffres.")


if __name__ == "__main__":
    main()

