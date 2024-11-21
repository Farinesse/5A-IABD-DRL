import numpy as np
import random
from gymnasium import spaces
from tensorflow.python import keras
from typing import List, Tuple, Dict, Optional, Any


class FarkleGame:
    """Base class for the Farkle dice game."""

    def __init__(self, num_players: int = 2, target_score: int = 10000):
        """Initialize the Farkle game.

        Args:
            num_players: Number of players in the game
            target_score: Score required to win the game
        """
        self.num_players = int(num_players)
        self.target_score = target_score
        self._initialize_spaces()
        self.reset()

    def _initialize_spaces(self) -> None:
        """Initialize the observation and action spaces."""
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] + [0] * self.num_players + [0] * 6 + [0]),
            high=np.array([self.num_players - 1, self.target_score, 6] +
                          [self.target_score] * self.num_players + [6] * 6 + [1]),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(128)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the game state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation and empty info dict
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.scores = [0] * self.num_players
        self.current_player = 0
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = self._roll_dice(self.remaining_dice)
        self.game_over = False
        self.last_action_stop = False

        return self.get_observation(), {}

    def _roll_dice(self, num_dice: int) -> List[int]:
        """Roll the specified number of dice.

        Args:
            num_dice: Number of dice to roll

        Returns:
            List of dice values
        """
        return [random.randint(1, 6) for _ in range(num_dice)]

    def get_observation(self) -> np.ndarray:
        """Get the current game state observation."""
        padded_dice = self.dice_roll + [0] * (6 - len(self.dice_roll))
        return np.array([
                            self.current_player,
                            self.round_score,
                            self.remaining_dice
                        ] + self.scores + padded_dice + [int(self.last_action_stop)])

    def state_description(self) -> np.ndarray:
        """Get normalized state description for RL agent."""
        state = np.zeros(12)
        state[0] = self.current_player
        state[1] = self.round_score / self.target_score
        state[2] = self.remaining_dice / 6
        state[3:5] = [score / self.target_score for score in self.scores]
        state[5:11] = [roll / 6 for roll in self.dice_roll] + [0] * (6 - len(self.dice_roll))
        state[11] = int(self.last_action_stop)
        return state

    def step(self, action: List[int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute a game step with the given action.

        Args:
            action: Binary action list (dice selection + stop bit)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Process current player's action
        result = self._process_player_action(action)

        # If game not over and it's player 2's turn, make them play immediately
        if not self.game_over and self.current_player == 1:
            valid_actions = self.available_actions_ids()
            ai_action = self.decode_action(np.random.choice(valid_actions))
            result = self._process_player_action(ai_action)

        return result

    def _process_player_action(self, action: List[int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Process a single player's action."""
        kept_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action[i] == 1]

        # Check for hot dice (all dice scoring)
        new_score = self._calculate_score(self.dice_roll, False)
        if new_score == 0 and self.remaining_dice == 6:
            self.round_score += 500
            self.next_player()
            return self.get_observation(), 500, False, False, {"stopped": True}

        # Calculate score for kept dice
        new_score = self._calculate_score(kept_dice, True)
        if new_score == 0:
            lost_points = self.round_score
            self.round_score = 0
            self.next_player()
            return self.get_observation(), -lost_points, False, False, {"farkle": True, "lost_points": lost_points}

        # Update game state
        self.round_score += new_score
        self.remaining_dice -= sum(action[:len(self.dice_roll)])

        if self.remaining_dice == 0:
            self.remaining_dice = 6

        # Handle stopping
        self.last_action_stop = bool(action[-1])
        if self.last_action_stop:
            self.scores[self.current_player] += self.round_score
            reward = self.round_score

            if self.scores[self.current_player] >= self.target_score:
                self.game_over = True
                return self.get_observation(), reward, True, False, {"win": True}

            self.next_player()
            return self.get_observation(), reward, False, False, {"stopped": True}

        self.dice_roll = self._roll_dice(self.remaining_dice)
        return self.get_observation(), new_score, False, False, {}

    def _calculate_score(self, dice_roll: List[int], use_restriction: bool = True) -> int:
        """Calculate score for selected dice."""
        if not dice_roll:
            return 0

        counts = [dice_roll.count(i) for i in range(1, 7)]

        # Special combinations
        if sorted(dice_roll) == [1, 2, 3, 4, 5, 6]:
            return 1500
        if counts.count(2) == 3:
            return 1000

        score = 0
        # Score for sets of 3 or more
        for die in range(3, 7):
            if die in counts:
                for i, count in enumerate(counts):
                    if count == die:
                        coef = 1000 if (i + 1) == 1 else 100
                        score += (i + 1) * coef * 2 ** (die - 3)

        # Score for single 1s and 5s
        score += counts[0] * 100 if counts[0] < 3 else 0
        score += counts[4] * 50 if counts[4] < 3 else 0

        # Apply restrictions if needed
        if use_restriction:
            for i in [1, 2, 3, 5]:
                if 0 < counts[i] < 3:
                    return 0

        return score

    def next_player(self) -> None:
        """Switch to the next player."""
        self.current_player = (self.current_player + 1) % self.num_players
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = self._roll_dice(self.remaining_dice)
        self.last_action_stop = False

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.game_over


class FarkleDQNEnv(FarkleGame):
    """Environment wrapper for DQN training."""

    def __init__(self, num_players: int = 2, target_score: int = 5000):
        super().__init__(num_players, target_score)
        self.action_space_size = 128

    def available_actions_ids(self) -> np.ndarray:
        """Get valid action IDs."""
        valid_mask = self._get_valid_actions()
        return np.where(valid_mask == 1)[0]

    def action_mask(self) -> np.ndarray:
        """Get binary mask for valid actions."""
        mask = np.zeros(128, dtype=np.float32)
        mask[self.available_actions_ids()] = 1.0
        return mask

    def decode_action(self, action_id: int) -> List[int]:
        """Convert action ID to binary action list."""
        return [int(b) for b in format(action_id, '07b')]

    def score(self) -> float:
        """Get the current game score."""
        if self.game_over:
            if self.scores[0] >= self.target_score:
                return 1.0
            elif self.scores[1] >= self.target_score:
                return -1.0
        return 0.0

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.game_over
    def get_random_action(self) -> int:
        """Get a random valid action ID."""
        valid_actions = self.available_actions_ids()
        return np.random.choice(valid_actions)
    def display(self) -> None:
        """Display current game state."""
        print(f"\nCurrent Player: {self.current_player + 1}")
        print(f"Round Score: {self.round_score}")
        print(f"Remaining Dice: {self.remaining_dice}")
        print(f"Current Roll: {self.dice_roll}")
        print(f"Player Scores: {self.scores}")
        print(f"Game Over: {self.game_over}")


def create_farkle_model() -> keras.Sequential:
    """Create neural network model for Farkle."""
    return keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_dim=12),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128)
    ])


if __name__ == "__main__":
    # Test game
    env = FarkleDQNEnv()
    state = env.reset()

    while not env.is_game_over():
        env.display()
        action_id = env.get_random_action()
        state, reward, done, _, info = env.step(action_id)
        print(f"Reward: {reward}")
        if info.get("win"):
            print(f"Player {env.current_player + 1} wins!")