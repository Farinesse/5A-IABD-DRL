import random
import numpy as np
import time

import numpy as np
from tqdm import tqdm


class FarkleEnv:
    def __init__(self, num_players=1, target_score=2000):
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

    # Modification de la méthode step dans la classe FarkleEnv
    def step(self, action):
        """Exécute une action dans l'environnement."""
        if len(action) != 7:
            raise ValueError("Action invalide. Vous devez entrer une séquence binaire de 7 chiffres.")

        initial_score = self.calculate_score(self.dice_roll, False)

        # Bonus de 500 points pour un Farkle avec tous les dés
        if initial_score == 0 and self.remaining_dice == 6:
            self.round_score += 500
            self.next_player()
            return self.get_observation(), 500, False, {"farkle": True}

        # Validation de la sélection des dés
        dice_action = action[:len(self.dice_roll)]
        if not validate_dice_selection(self.dice_roll, dice_action):
            self.round_score = 0
            self.next_player()
            return self.get_observation(), -1, False, {"invalid_selection": True}

        # Sélection des dés gardés
        kept_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action[i] == 1]
        new_score = self.calculate_score(kept_dice)

        # Mise à jour du nombre de dés restants
        self.remaining_dice -= sum(dice_action)

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



def random_agent(env, dice_roll):
    """Agent aléatoire qui choisit une action binaire aléatoire de 7 chiffres."""
    action = [random.choice([0, 1]) for _ in range(6)]  # Pour les 6 dés
    action.append(random.choice([0, 1]))  # Pour l'action "stop"
    return action



def validate_dice_selection(dice_roll, action):
    """Valide que seuls les dés ayant une valeur peuvent être sélectionnés."""
    counts = [dice_roll.count(i) for i in range(1, 7)]
    valid_singles = {1, 5}  # Dés qui peuvent être sélectionnés individuellement

    selected_dice = [d for i, d in enumerate(dice_roll) if action[i] == 1]
    if not selected_dice:
        return False

    # Vérification de la suite
    if sorted(selected_dice) == [1, 2, 3, 4, 5, 6]:
        return True

    # Vérification des trois paires
    selected_counts = [selected_dice.count(i) for i in range(1, 7)]
    if len(selected_dice) == 6 and selected_counts.count(2) == 3:
        return True

    # Vérification des sélections invalides
    for value, count in enumerate(selected_counts, start=1):
        if count > 0:  # Si le dé est sélectionné
            original_count = dice_roll.count(value)
            if value not in valid_singles and count < 3 and original_count < 3:
                # On ne peut pas sélectionner moins de 3 dés pour les valeurs autres que 1 et 5
                return False

    return True

def human_agent(env, dice_roll):
    """Agent humain pour choisir les dés à garder ou relancer."""
    print(f"Résultat des dés actuels : {dice_roll}")
    print("Rappel des règles :")
    print("- Vous pouvez sélectionner les 1 et les 5 individuellement")
    print("- Pour les autres chiffres, vous devez sélectionner au moins trois dés identiques")
    print("- Vous pouvez sélectionner une suite (1-2-3-4-5-6)")
    print("- Vous pouvez sélectionner trois paires")

    while True:
        try:
            action_str = input(
                f"Entrez une action binaire de 7 chiffres (0 = relancer, 1 = garder) et le dernier chiffre pour arrêter le tour : ")
            if len(action_str) == 7 and all(c in '01' for c in action_str):
                action = [int(x) for x in action_str]
                if validate_dice_selection(dice_roll, action[:len(dice_roll)]):
                    return action
                else:
                    print("Sélection invalide. Veuillez respecter les règles du jeu.")
            else:
                print("Action invalide. Veuillez entrer une séquence binaire de 7 chiffres.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer une séquence binaire de 7 chiffres.")


def random_agent(env, dice_roll):
    """Agent aléatoire qui choisit une action binaire aléatoire de 7 chiffres."""
    action = [random.choice([0, 1]) for _ in range(6)]  # Pour les 6 dés
    action.append(random.choice([0, 1]))  # Pour l'action "stop"
    return action


def play_one_game(env):
    env.reset()
    done = False
    while not done:
        while True:
            env.dice_roll = env.roll_dice(env.remaining_dice)
            action = random_agent(env, env.dice_roll)
            _, _, done, _ = env.step(action)
            if done or env.last_action_stop or env.remaining_dice == 0:
                break
    return max(env.scores)  # Retourne le score gagnant


def simulate_games_for_30_seconds(num_players=2, target_score=10000):
    env = FarkleEnv(num_players=num_players, target_score=target_score)
    start_time = time.time()
    total_score = 0
    num_games = 0

    with tqdm(total=30, desc="Simulation en cours", unit="s") as pbar:
        while time.time() - start_time < 30:
            winning_score = play_one_game(env)
            total_score += winning_score
            num_games += 1

            # Mise à jour de la barre de progression
            pbar.update(time.time() - start_time - pbar.n)

    end_time = time.time()
    total_elapsed_time = end_time - start_time
    games_per_second = num_games / total_elapsed_time
    average_score = total_score / num_games

    return num_games, games_per_second, average_score


def main():
    print("Simulation de parties de Farkle pendant 30 secondes...")
    num_games, games_per_second, average_score = simulate_games_for_30_seconds()
    print(f"\nRésultats après 30 secondes de simulation :")
    print(f"Nombre de parties jouées : {num_games}")
    print(f"Parties par seconde : {games_per_second:.2f}")
    print(f"Score moyen : {average_score:.2f}")


if __name__ == "__main__":
    main()