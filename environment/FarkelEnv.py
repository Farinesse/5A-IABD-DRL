import numpy as np
import random
from gymnasium import spaces


class FarkleEnv:
    def __init__(self, num_players=2, target_score=10000):
        self.num_players = num_players
        self.target_score = target_score

        # Définition des espaces d'observation et d'action
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] + [0] * num_players + [0] * 6 + [0]),
            high=np.array([num_players - 1, target_score, 6] + [target_score] * num_players + [6] * 6 + [1]),
            dtype=np.int32
        )

        self.action_space = spaces.Discrete(128)  # 2^7 possibilités pour les 7 bits
        self.reset()

    def describe_action(action):
        """Décrit l'action en termes de quels dés sont gardés."""
        binary = format(action, '07b')  # Convertir en chaîne binaire sur 7 bits
        dice_kept = [i + 1 for i, bit in enumerate(binary) if bit == '1']
        return dice_kept

    def reset(self, seed=None):
        """Réinitialise l'environnement pour un nouvel épisode."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.scores = [0] * self.num_players
        self.current_player = 0
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = self.roll_dice(self.remaining_dice)  # Lancer les dés au début
        self.game_over = False
        self.last_action_stop = False
        return self.get_observation(), {}

    def get_observation(self):
        """Retourne l'observation de l'état actuel."""
        padded_dice = self.dice_roll + [0] * (6 - len(self.dice_roll))
        obs = np.array([self.current_player, self.round_score, self.remaining_dice] + self.scores + padded_dice + [int(self.last_action_stop)])
        return obs

    def get_valid_actions(self):
        """Retourne un masque des actions valides."""
        if not self.dice_roll:
            return np.zeros(128, dtype=np.int8)

        valid_mask = np.zeros(128, dtype=np.int8)
        for action in range(128):
            binary = format(action, '07b')
            action_list = [int(b) for b in binary]
            if self._validate_dice_selection(self.dice_roll, action_list[:len(self.dice_roll)]):
                valid_mask[action] = 1
        return valid_mask

    def _validate_dice_selection(self, dice_roll, action):
        """Valide que seuls les dés ayant une valeur peuvent être sélectionnés."""
        if len(action) < len(dice_roll):
            return False

        selected_dice = [d for i, d in enumerate(dice_roll) if action[i] == 1]
        if not selected_dice:
            return True  # Permettre de ne rien sélectionner

        # Vérification de la suite
        if sorted(selected_dice) == [1, 2, 3, 4, 5, 6]:
            return True

        # Vérification des trois paires
        selected_counts = [selected_dice.count(i) for i in range(1, 7)]
        if len(selected_dice) == 6 and selected_counts.count(2) == 3:
            return True

        # Vérification des sélections valides
        valid_singles = {1, 5}
        for value, count in enumerate(selected_counts, start=1):
            if count > 0:
                original_count = dice_roll.count(value)
                if value not in valid_singles and count < 3 and original_count < 3:
                    return False

        return True

    def _calculate_score(self, dice_roll):
        """Calcule les points selon les dés lancés."""
        if not dice_roll:
            return 0

        counts = [dice_roll.count(i) for i in range(1, 7)]
        score = 0

        if sorted(dice_roll) == [1, 2, 3, 4, 5, 6]:
            return 1500
        if len(dice_roll) == 6 and counts.count(2) == 3:
            return 1000

        for value, count in enumerate(counts, start=1):
            if count >= 3:
                if value == 1:
                    score += 1000 * (2 ** (count - 3))
                else:
                    score += value * 100 * (2 ** (count - 3))

        ones_count = counts[0] % 3
        fives_count = counts[4] % 3
        score += ones_count * 100 + fives_count * 50

        return score

    def step(self, action):
        """Exécute une action dans l'environnement."""
        binary = format(action, '07b')
        action_list = [int(b) for b in binary]

        if not self._validate_dice_selection(self.dice_roll, action_list[:len(self.dice_roll)]):
            print("Action invalide : La sélection des dés n'est pas valide.")
            return self.get_observation(), -100, True, False, {"invalid_action": True}

        kept_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action_list[i] == 1]
        print(f"Dés sélectionnés : {kept_dice}")
        new_score = self._calculate_score(kept_dice)
        print(f"Points gagnés cette action : {new_score}")

        # Gestion des cas spéciaux : si aucun point n'est gagné
        if new_score == 0:
            self.round_score = 0
            self.next_player()  # Passer au joueur suivant ou terminer le tour
            self.dice_roll = self.roll_dice(self.remaining_dice)  # Relancer les dés
            return self.get_observation(), -50, False, False, {"farkle": True}

        self.round_score += new_score
        self.remaining_dice -= sum(action_list[:len(self.dice_roll)])

        # Si aucun dé n'est resté, relancer 6 dés
        if self.remaining_dice == 0:
            self.remaining_dice = 6

        # Mise à jour des scores du joueur actuel
        self.scores[self.current_player] += new_score

        # Vérification de la victoire après chaque action
        if self.scores[self.current_player] >= self.target_score:
            self.game_over = True
            return self.get_observation(), new_score, True, False, {"win": True}

        # Gestion de l'action "stop" (si le joueur décide d'arrêter)
        self.last_action_stop = bool(action_list[-1])
        if self.last_action_stop:
            reward = self.round_score
            self.next_player()  # Passer au joueur suivant
            self.dice_roll = self.roll_dice(self.remaining_dice)  # Relancer les dés pour le prochain joueur
            return self.get_observation(), reward, False, False, {"stopped": True}

        # Si le joueur n'a pas arrêté, relancer les dés restants
        self.dice_roll = self.roll_dice(self.remaining_dice)
        return self.get_observation(), new_score, False, False, {}

    def roll_dice(self, num_dice):
        """Lance un nombre donné de dés."""
        return [random.randint(1, 6) for _ in range(num_dice)]

    def next_player(self):
        """Passe au joueur suivant."""
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


def main():
    # Initialisation de l'environnement Farkle avec 2 joueurs
    env = FarkleEnv(num_players=2, target_score=10000)
    player_names = ["Joueur 1", "Joueur 2"]

    done = False  # Variable pour suivre la fin du jeu

    # Tant que la partie n'est pas terminée
    while not done:
        # Rendu de l'état initial
        env.render()

        # Tour du joueur courant
        current_player = player_names[env.current_player]
        print(f"\nTour de {current_player} :")
        valid_actions = env.get_valid_actions()
        valid_indices = np.where(valid_actions == 1)[0]

        if len(valid_indices) == 0:
            print("Aucune action valide disponible, tour terminé.")
            observation, reward, done, _, info = env.step(0)  # Action par défaut
        else:
            # Afficher les actions valides en binaire
            valid_actions_binary = [format(action, '07b') for action in valid_indices]
            print(f"Actions valides (binaire) : {list(zip(valid_indices, valid_actions_binary))}")

            # L'utilisateur choisit une action en binaire ou en entier
            user_input = input(f"Choisissez une action parmi {valid_actions_binary} (en binaire ou entier): ")

            # Vérifie si l'utilisateur a entré un nombre binaire ou un entier
            if all(char in '01' for char in user_input):
                action = int(user_input, 2)  # Convertir l'entrée binaire en entier
            else:
                action = int(user_input)  # Traiter comme un entier

            # Validation de l'action
            while action not in valid_indices:
                user_input = input(f"Action invalide. Choisissez parmi {valid_actions_binary}: ")
                if all(char in '01' for char in user_input):
                    action = int(user_input, 2)  # Convertir l'entrée binaire en entier
                else:
                    action = int(user_input)  # Traiter comme un entier

            # Exécuter l'action choisie
            observation, reward, done, _, info = env.step(action)

            # Affichage des résultats du tour
            print(f"Récompense obtenue : {reward}")
            env.render()

            # Si l'utilisateur décide d'arrêter ou fait un Farkle, terminer le tour
            if info.get("stopped", False):
                print(f"{current_player} a arrêté le tour.")
            elif info.get("farkle", False):
                print(f"{current_player} a fait un Farkle ! Aucun point gagné.")

        # Vérification si un joueur a gagné
        if reward == 1000:
            print(f"{current_player} a gagné la partie !")
            break

    # Fin de la partie après un tour
    print("\nFin de la partie !")


if __name__ == "__main__":
    main()
