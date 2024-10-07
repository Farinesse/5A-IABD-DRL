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

class FarkleDQNEnv(FarkleEnv):
    def __init__(self, num_players=1, target_score=10000):
        super().__init__(num_players, target_score)
        self.action_space_size = 128  # 2^7 possibilités (6 dés + stop action)

    def action_mask(self):
        """Crée un masque pour les actions valides de manière optimisée."""
        mask = np.zeros(self.action_space_size, dtype=np.int8)

        if not self.dice_roll:
            mask[127] = 1  # Permet toujours l'action "stop" quand pas de dés
            return mask

        for i in range(self.action_space_size):
            binary = format(i, '07b')
            action = [int(b) for b in binary]
            current_action = action[:len(self.dice_roll)]

            # Vérification rapide si l'action est valide
            if action[-1] == 1:  # Action "stop" toujours valide
                mask[i] = 1
                continue

            kept_dice = [d for j, d in enumerate(self.dice_roll) if current_action[j] == 1]
            if self.calculate_score(kept_dice) > 0:
                mask[i] = 1

        return mask

    def available_actions_ids(self):
        # Cette fonction devrait toujours retourner au moins une action valide
        # Par exemple, l'action de passer son tour devrait toujours être disponible
        actions = [i for i, mask in enumerate(self.action_mask()) if mask > 0]
        if not actions:
            return np.array([127])  # Action par défaut si aucune autre n'est disponible
        return np.array(actions)

    def decode_action(self, action_id):
        """Décode l'ID d'action en une liste d'actions binaires."""
        binary = format(action_id, '07b')  # Toujours 7 bits pour 6 dés + action stop
        return [int(b) for b in binary]

    def step(self, action_id):
        """Exécute une action dans l'environnement."""
        action = self.decode_action(action_id)
        return super().step(action)
