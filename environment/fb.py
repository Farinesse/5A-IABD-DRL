import numpy as np

from rand import random


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
        self.last_action_stop = False  # Changed to boolean
        return self.get_observation()

    def get_observation(self):
        """Retourne une observation de l'état actuel sous forme de vecteur."""
        obs = np.array([self.current_player, self.round_score, self.remaining_dice] +
                       self.scores + self.dice_roll + [int(self.last_action_stop)])
        return obs

    def roll_dice(self, num_dice):
        """Lance un nombre donné de dés."""
        self.dice_roll = [random.randint(1, 6) for _ in range(num_dice)]
        return self.dice_roll

    def calculate_score(self, dice_roll, use_restriction = True):
        """Calcule les points selon les dés lancés."""

        if not dice_roll:
            return 0

        counts = [dice_roll.count(i) for i in range(1, 7)]
        score = 0
        if sorted(dice_roll) == [1, 2, 3, 4, 5, 6]:
            return 1500

        if counts.count(2) == 3:
            return 1000

        for die in range(3, 7):
            if die in counts:
                for i,num in enumerate(counts):
                    if counts[i] == die:
                        coef = 1000 if (i+1) == 1 else 100
                        score += (i+1) * coef * 2 ** (die - 3)

        score += counts[0] * 100 if counts[0] < 3 else 0
        score += counts[4] * 50 if counts[4] < 3 else 0

        if  use_restriction:
            score = 0 if counts[1] < 3 and counts[1] != 0 else score
            score = 0 if counts[2] < 3 and counts[2] != 0 else score
            score = 0 if counts[3] < 3 and counts[3] != 0 else score
            score = 0 if counts[5] < 3 and counts[5] != 0 else score

        return score

    def step(self, action):
        """Exécute une action dans l'environnement."""
        if len(action) != 7:  # Toujours 6 dés + l'action de stop
            raise ValueError(f"Action invalide. Vous devez entrer une séquence binaire de 7 chiffres.")

        score = self.calculate_score(self.dice_roll,False)
        if score == 0 and self.remaining_dice == 6:
            self.round_score = self.round_score + 500
            self.next_player()
            return self.get_observation(), self.round_score, False, {"farkle": True}

        kept_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action[i] == 1]
        score = self.calculate_score(kept_dice)
        self.remaining_dice = len([self.dice_roll[i] for i in range(len(self.dice_roll)) if action[i] == 0])

        if score == 0 and  self.remaining_dice < 6:
            self.round_score = 0
            self.next_player()
            return self.get_observation(), -1, False, {"farkle": True}

        self.round_score += score

        done = self.scores[self.current_player] + self.round_score >= self.target_score

        if done:
            self.scores[self.current_player] += self.round_score
            self.game_over = True
            return self.get_observation(), score, True, {"win": True}

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
    env = FarkleEnv(num_players=2)
    env.reset()

    while not env.is_game_over():
        print(f"\nTour du joueur {env.current_player + 1}...\n")

        while True:
            dice_roll = env.roll_dice(env.remaining_dice)
            env.render()
            score = env.calculate_score(env.dice_roll, False )
            print("avant step", score, env.remaining_dice, env.dice_roll)
            if score == 0 and env.remaining_dice < 6 :

                print("arret", score,env.remaining_dice )
                observation, reward, done, _ = env.get_observation(), -1, False, {"farkle": True}
                env.last_action_stop = True
                env.round_score = 0
                env.next_player()

            else :
                if env.current_player == 0:
                    action = human_agent(env, dice_roll)
                else:
                    action = random_agent(env, dice_roll)
                observation, reward, done, _ = env.step(action)


            print(f"Récompense: {reward}")

            if done:
                print(f"Le joueur {env.current_player + 1} a terminé le jeu !")
                break

            if env.last_action_stop or env.remaining_dice == 0:
                break

        if done:
            break

    print(f"Scores finaux: {env.scores}")
    print(f"Le joueur {env.current_player + 1} a gagné!")