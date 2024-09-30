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
        # Normaliser les valeurs pour une meilleure stabilité d'apprentissage
        normalized_score = self.round_score / self.target_score
        normalized_total_score = self.scores[self.current_player] / self.target_score

        # Créer un vecteur one-hot pour les dés
        dice_state = np.zeros(6)
        for die in self.dice_roll:
            dice_state[die - 1] += 1
        dice_state = dice_state / 6  # Normaliser

        state = np.array([
            normalized_score,
            normalized_total_score,
            self.remaining_dice / 6,
            *dice_state,  # Décompresser le vecteur des dés
            int(self.last_action_stop)
        ])
        return state

    def roll_dice(self, num_dice):
        """Lance un nombre donné de dés."""
        self.dice_roll = [random.randint(1, 6) for _ in range(num_dice)]
        return self.dice_roll

    def calculate_score(self, dice_roll):
        """Calcule les points selon les dés lancés."""
        if not dice_roll:
            return 0

        counts = [dice_roll.count(i) for i in range(1, 7)]
        score = 0
        score += counts[0] * 100  # 1 vaut 100 points
        score += counts[4] * 50  # 5 vaut 50 points

        for i in range(6):
            if counts[i] >= 3:
                score += (i + 1) * 100 if i > 0 else 1000  # Trois dés identiques
        return score

    def step(self, action):
        """Exécute une action dans l'environnement."""
        if len(action) != len(self.dice_roll) + 1:
            raise ValueError(
                f"Action invalide. Vous devez entrer une séquence binaire de {len(self.dice_roll) + 1} chiffres.")

        kept_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action[i] == 1]
        remaining_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action[i] == 0]

        score = self.calculate_score(kept_dice)

        if score == 0 and action[-1] == 0:  # Farkle (pas de points valides)
            self.round_score = 0
            self.next_player()
            return self.get_observation(), -1, False, {}

        self.round_score += score
        self.remaining_dice = len(remaining_dice) if remaining_dice else 6

        done = self.scores[self.current_player] + self.round_score >= self.target_score
        if done:
            self.scores[self.current_player] += self.round_score
            self.game_over = True
            return self.get_observation(), score, True, {}

        self.last_action_stop = bool(action[-1])
        if self.last_action_stop:
            self.next_player()

        return self.get_observation(), score, False, {}

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
        print(f"Score du tour: {self.round_score}, Dés restants: {self.remaining_dice}")
        print(f"Scores actuels: {self.scores}")

    def is_game_over(self):
        """Vérifie si la partie est terminée."""
        return self.game_over


class FarkleDQNEnv(FarkleEnv):
    def __init__(self, num_players=1, target_score=10000):
        super().__init__(num_players, target_score)
        self.action_space_size = 128  # 2^7 possibilités (6 dés + stop action)

    def action_mask(self):
        """Crée un masque pour les actions valides."""
        mask = np.zeros(self.action_space_size)

        if not self.dice_roll:
            return mask

        # Générer toutes les combinaisons possibles d'actions
        for i in range(self.action_space_size):
            binary = format(i, f'0{len(self.dice_roll) + 1}b')
            action = [int(b) for b in binary]

            if len(action) == len(self.dice_roll) + 1:
                # Vérifier si l'action est valide
                kept_dice = [self.dice_roll[j] for j in range(len(self.dice_roll)) if action[j] == 1]
                if self.calculate_score(kept_dice) > 0 or action[-1] == 1:
                    mask[i] = 1

        return mask

    def available_actions_ids(self):
        """Retourne les IDs des actions disponibles."""
        return np.where(self.action_mask() == 1)[0]

    def decode_action(self, action_id):
        """Décode l'ID d'action en une liste d'actions binaires."""
        binary = format(action_id, f'0{len(self.dice_roll) + 1}b')
        return [int(b) for b in binary]

    def step(self, action_id):
        """Exécute une action dans l'environnement."""
        action = self.decode_action(action_id)
        return super().step(action)