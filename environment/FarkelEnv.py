import numpy as np
import random
from gymnasium import spaces

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
        padded_dice = self.dice_roll + [0] * (6 - len(self.dice_roll))
        return np.array([self.current_player, self.round_score, self.remaining_dice] +
                        self.scores + padded_dice + [int(self.last_action_stop)])

    def roll_dice(self, num_dice):
        return [random.randint(1, 6) for _ in range(num_dice)]

    def get_valid_actions(self):
        valid_mask = np.zeros(128, dtype=np.int8)
        for action in range(128):
            binary = format(action, '07b')
            action_list = [int(b) for b in binary]

            if self._validate_dice_selection(self.dice_roll, action_list[:len(self.dice_roll)]):
                valid_mask[action] = 1
                print(action)

        return valid_mask

    def _validate_dice_selection(self, dice_roll, action):

        if len(action) < len(dice_roll):
            return False

        selected_dice = [d for i, d in enumerate(dice_roll) if action[i] == 1]
        if not selected_dice:
            return True

        if sorted(selected_dice) == [1, 2, 3, 4, 5, 6]:
            return True

        if (self.stop or action == [1,1,1,1,1,1,1]) :
            print(self.stop)
            return True

        selected_counts = [selected_dice.count(i) for i in range(1, 7)]
        if len(selected_dice) == 6 and selected_counts.count(2) == 3:
            return True

        valid_singles = {1, 5}
        for value, count in enumerate(selected_counts, start=1):
            if count > 0:
                original_count = dice_roll.count(value)
                if value not in valid_singles and count < 3 and original_count < 3:
                    return False

        return True

    def _calculate_score(self, dice_roll, use_restriction = True):


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

    def step(self, action):
        if self.current_player == 1:
            action = self.get_random_action()

        print(self.current_player)
        print(self.dice_roll)
        print(self.get_valid_actions())
        action_list = action
        print(action)
        print("**************************************************************************")
        if not self._validate_dice_selection(self.dice_roll, action_list[:len(self.dice_roll)]):
            return self.get_observation(), -100, True, False, {"invalid_action": True}

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

    def get_random_action(self):
        valid_actions = self.get_valid_actions()
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) > 0:
            random_action = random.choice(valid_indices)
            return [int(b) for b in format(random_action, '07b')]
        else:
            return [0] * 6 + [1]