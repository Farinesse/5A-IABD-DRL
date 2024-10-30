import numpy as np
import random
from gymnasium import spaces
from tensorflow.python import keras
#from algos.DQN.ddqn import double_dqn_no_replay
from algos.DQN.deep_qlearning import deep_q_learning


def play_game_manual():
    """Fonction pour jouer manuellement contre un adversaire aléatoire."""
    env = FarkleDQNEnv(num_players=2, target_score=5000)
    state, _ = env.reset()
    done = False

    while not env.is_game_over():
        # Affichage plus clair de l'état du jeu
        print("\n" + "=" * 50)
        print("État du jeu:")
        print(f"🎲 Dés actuels: {env.dice_roll}")
        print(f"🎯 Score du tour: {env.round_score}")
        print(f"👥 Scores des joueurs: {env.scores}")
        print(f"🎮 Joueur actuel: {env.current_player + 1}")
        print(f"🎲 Dés restants: {env.remaining_dice}")
        print("=" * 50 + "\n")

        if env.current_player == 0:  # Tour du joueur humain
            # Affichage des actions valides
            print("\nActions valides disponibles:")
            valid_actions = env.available_actions_ids()
            for action_id in valid_actions:
                action_binary = format(action_id, '07b')
                print(f"\nID: {action_id}")
                print(f"Action binaire: {action_binary}")
                # Explication détaillée de l'action
                dice_selection = list(action_binary[:-1])
                stop_action = action_binary[-1]

                # Montrer quels dés seraient gardés
                kept_dice = []
                for i, (die, keep) in enumerate(zip(env.dice_roll, dice_selection)):
                    if keep == '1':
                        kept_dice.append(die)

                print(f"Dés à garder: {kept_dice}")
                print(f"Action stop: {'Oui' if stop_action == '1' else 'Non'}")

            # Saisie de l'action avec validation
            while True:
                try:
                    action = int(input("\nEntrez l'ID de votre action : "))
                    if action_id in valid_actions:
                    #    action = [int(b) for b in format(action_id, '07b')]
                       break
                    print("❌ Action invalide. Veuillez choisir parmi les actions listées.")
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide.")

        else:  # Tour de l'adversaire aléatoire
            print("\nTour de l'adversaire...")
            action = env.get_random_action()
            print(f"L'adversaire choisit : {action}")

        # Exécution de l'action
        state, reward, done, truncated, info = env.step(action)

        # Affichage des résultats
        if info.get("farkle"):
            print(f"\n🎲 FARKLE! Perte de {info['lost_points']} points")
        elif info.get("invalid_action"):
            print("\n❌ Action invalide!")
        elif info.get("stopped"):
            print(f"\n🛑 Tour terminé! Points gagnés: {reward}")
        elif info.get("win"):
            print(f"\n🏆 Victoire! Points finaux: {reward}")
        else:
            print(f"\n✔️ Points gagnés ce coup: {reward}")

    # Affichage des résultats finaux
    print("\n🎮 Partie terminée!")
    print(f"Scores finaux: Joueur 1 = {env.scores[0]}, Joueur 2 = {env.scores[1]}")
    if env.scores[0] > env.scores[1]:
        print("🎉 Félicitations! Vous avez gagné!")
    else:
        print("😔 L'adversaire a gagné. Meilleure chance la prochaine fois!")


class FarkleEnv:
    def __init__(self, num_players=2, target_score=10000):
        self.num_players = int(num_players)
        self.target_score = target_score
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] + [0] * self.num_players + [0] * 6 + [0]),
            high=np.array([self.num_players - 1, target_score, 6] + [target_score] * self.num_players + [6] * 6 + [1]),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(128)
        self.reset()
        self.stop = False

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.scores = [0] * self.num_players
        self.current_player = 0
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = self.roll_dice(self.remaining_dice)
        self.game_over = False
        self.last_action_stop = False
        return self.get_observation(), {}
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

    def state_description(self) -> np.ndarray:
        """
        Encode l'état du jeu dans un vecteur de taille fixe (12).
        """

        current_state = np.zeros(12)

        current_state[0] = self.current_player
        current_state[1] = self.round_score / self.target_score
        current_state[2] = self.remaining_dice / 6
        current_state[3:5] = [score / self.target_score for score in self.scores]  # Normalisation élément par élément
        #current_state[5:11] = self.dice_roll + [0] * (6 - len(self.dice_roll))
        current_state[5:11] = [roll / 6 for roll in self.dice_roll] + [0] * (6 - len(self.dice_roll))
        current_state[11] = int(self.last_action_stop)

        return current_state

    def roll_dice(self, num_dice):
        return [random.randint(1, 6) for _ in range(num_dice)]

    def get_valid_actions(self):
        valid_mask = np.zeros(128, dtype=np.int8)
        has_valid_action = False

        for action in [i for i in range(0, 128, 2)]:
            binary = format(action, '07b')
            action_list = [int(b) for b in binary]
            if self._validate_dice_selection(self.dice_roll,
                                             action_list[:len(self.dice_roll)] + [0] * (6 - len(self.dice_roll)) + [
                                                 action_list[-1]]):
                valid_mask[action] = 1
                has_valid_action = True

        for i, vm in enumerate(valid_mask[::-1]):
            if vm == 1:
                valid_mask[-i] = 1

                break

        if valid_mask[sum([2 ** (6 - i) for i in range(self.remaining_dice)])] == 1:
            valid_mask = np.zeros(128, dtype=np.int8)
            valid_mask[sum([2 ** (6 - i) for i in range(self.remaining_dice)])] = 1

        valid_mask[-1] = 0

        if not has_valid_action:
            default_action = int('0000001', 2)
            valid_mask[default_action] = 1
        return valid_mask

    def _validate_dice_selection(self, dice_roll, action):

        if len(action) < len(dice_roll):
            return False

        if action == [1] * 7:
            return False

        for d, a in zip(dice_roll, action[:6]):
            if d == 0 and a == 1:
                return False

        selected_dice = [d for i, d in enumerate(dice_roll) if action[i] == 1]
        if not selected_dice:
            return False

        if sorted(selected_dice) == [1, 2, 3, 4, 5, 6]:
            return True

        selected_counts = [selected_dice.count(i) for i in range(1, 7)]

        if len(selected_dice) == 6 and selected_counts.count(2) == 3:
            return True

        valid_singles = {1, 5}

        f_counts = 0
        for value, count in enumerate(selected_counts, start=1):
            if count > 0:
                # original_count = dice_roll.count(value)
                if value not in valid_singles and count < 3:  # and original_count < 3:
                    f_counts += 1

        if f_counts > 0:
            if f_counts == (6 - selected_counts.count(0)) and (action[:6] == [1] * 6 or action[:6] == [0] * 6):
                return True
            else:
                return False
        else:
            return True

    def _calculate_score(self, dice_roll, use_restriction=True):

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
                for i, num in enumerate(counts):
                    if counts[i] == die:
                        coef = 1000 if (i + 1) == 1 else 100
                        score += (i + 1) * coef * 2 ** (die - 3)

        score += counts[0] * 100 if counts[0] < 3 else 0
        score += counts[4] * 50 if counts[4] < 3 else 0

        if use_restriction:
            score = 0 if counts[1] < 3 and counts[1] != 0 else score
            score = 0 if counts[2] < 3 and counts[2] != 0 else score
            score = 0 if counts[3] < 3 and counts[3] != 0 else score
            score = 0 if counts[5] < 3 and counts[5] != 0 else score

        return score

    def step(self, action ):


        #print(self.get_valid_actions())
        action_list = action
        #print(self.dice_roll)
        #print(action)#


        kept_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action_list[i] == 1]



        new_score = self._calculate_score(self.dice_roll, False)

        if new_score == 0 and self.remaining_dice == 6:
            self.round_score = self.round_score + 500
            self.next_player()
            return self.get_observation(), 500, False, False, {"stopped": True}

        new_score = self._calculate_score(kept_dice, not (self.stop))

        if new_score == 0:
            lost_points = self.round_score
            self.round_score = 0
            self.next_player()
            return self.get_observation(), -lost_points, False, False, {"farkle": True, "lost_points": lost_points}

        self.round_score += new_score
        self.remaining_dice -= sum(action_list[:len(self.dice_roll)])

        if self.remaining_dice == 0:
            self.remaining_dice = 6

        self.last_action_stop = bool(action_list[-1])
        if self.last_action_stop:
            self.scores[self.current_player] += self.round_score
            reward = self.round_score
            if self.scores[self.current_player] >= self.target_score:
                self.game_over = True
                return self.get_observation(), reward, True, False, {"win": True}
            self.next_player()
            return self.get_observation(), reward, False, False, {"stopped": True}

        self.dice_roll = self.roll_dice(self.remaining_dice)
        return self.get_observation(), new_score, False, False, {}


    def next_player(self):
        self.current_player = (self.current_player + 1) % self.num_players
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = self.roll_dice(self.remaining_dice)
        self.last_action_stop = False



    """def get_random_action(self):
        valid_actions = self.get_valid_actions()

        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) > 0:
            random_action = random.choice(valid_indices)
            return [int(b) for b in format(random_action, '07b')]
        else:
            return [0] * 6 + [1]"""

class FarkleDQNEnv(FarkleEnv):
    def __init__(self, num_players=2, target_score=1000):
        super().__init__(num_players, target_score)
        self.action_space_size = 128  # 2^7 possibilités (6 dés + stop action)
    def available_actions_ids(self) -> np.ndarray:
        """
        Retourne les indices des actions valides.
        """
        valid_mask = self.get_valid_actions()
        return np.where(valid_mask == 1)[0]

    def action_mask(self) -> np.ndarray:
        """Masque binaire pour les actions valides."""
        mask = np.zeros(128, dtype=np.float32)
        valid_actions = self.available_actions_ids()
        mask[valid_actions] = 1.0
        return mask

    def decode_action(self, action_id):
        """Conversion ID -> action binaire."""
        return [int(b) for b in format(action_id, '07b')]

    def score(self) -> float:

        # Si le jeu est terminé
        if self.game_over:
            if self.scores[0] >= self.target_score:
                return 1.0  # Victoire
            elif self.num_players > 1 and self.scores[1] >= self.target_score:
                return -1.0  # Défaite
            return 0.0  # Match nul ou autre condition

        # Si le jeu n'est pas terminé, vous pouvez choisir de retourner 0
        return 0.0

    def is_game_over(self) -> bool:
        """
        Indique si le jeu est terminé.
        """
        return self.game_over

    def decode_action(self, action_id):
        """Décode l'ID d'action en une liste d'actions binaires."""
        binary = format(action_id, '07b')  # Toujours 7 bits pour 6 dés + action stop
        return [int(b) for b in binary]

    def step(self, action_id):
        """Exécute une action cohérente selon l'ID."""
        if self.current_player == 1:
            # Pour le joueur aléatoire, utiliser directement un ID valide
            valid_actions = self.available_actions_ids()
            action_id = np.random.choice(valid_actions)

        # Convertir l'ID en action binaire
        action = self.decode_action(action_id)
        return super().step(action)

    def get_random_action(self):
        """Retourne un ID d'action valide."""
        valid_actions = self.available_actions_ids()
        return np.random.choice(valid_actions)

    def display(self):
        """
        Affiche l'état actuel du jeu.
        """
        print(f"Joueur actuel: {self.current_player + 1}")
        print(f"Score de la manche: {self.round_score}")
        print(f"Dés restants: {self.remaining_dice}")
        print(f"Dés actuels: {self.dice_roll}")
        print(f"Scores des joueurs: {self.scores}")
        print(f"Jeu terminé: {self.game_over}")



def create_farkle_model():
    """Crée le modèle pour Farkle avec la bonne taille d'entrée/sortie."""
    model = keras.Sequential([

        keras.layers.Dense(128, activation='relu', input_dim=12),  # 3 + num_players + 6 + 1
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128)  # Nombre d'actions possibles dans Farkle
    ])
    return model

if __name__ == "__main__":
    #play_game_manual()
    env = FarkleDQNEnv()

    model = create_farkle_model()
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    trained_model = deep_q_learning(
                model=model,
                target_model=target_model,
                env=env,
                num_episodes=20000,
                gamma=0.99,
                alpha=0.0001,
                start_epsilon=1.0,
                end_epsilon=0.01, #
                memory_size=512,
                batch_size=64,
                update_target_steps=500
            )
    '''
    final_online_model, final_target_model = double_dqn_no_replay(
        online_model=model,
        target_model=model,
        env=env,
        num_episodes=50000,
        gamma=0.99,
        alpha=0.0001,
        start_epsilon=1,
        end_epsilon=0.0001,
        update_target_steps=1000,
        save_path="ddqn_model_farkel_test2"

    )
    '''