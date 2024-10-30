import keras
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
from tqdm import tqdm  # Importer tqdm


class REINFORCE:
    def __init__(self, state_dim, action_dim, alpha=0.0003, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma

        # Construction du réseau de politique
        self.policy = self._build_policy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.reward_buffer = []

    def _build_policy(self):
        model = keras.Sequential([
            # Correction ici : enlever les parenthèses autour de state_dim
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_dim)
        ])
        return model

    def get_action(self, state, valid_actions):
        # Epsilon-greedy
        if np.random.random() < 0.05:
            return np.random.choice(valid_actions)
        # Forward pass
        logits = self.policy(state.reshape(1, -1), training=False)
        logits = logits.numpy()[0]

        # Masquer les actions invalides
        mask = np.ones_like(logits) * float('-inf')
        mask[valid_actions] = 0
        masked_logits = logits + mask

        # Softmax avec température
        temperature = max(0.5, 1.0 - len(self.reward_buffer) / 5000)
        probs = tf.nn.softmax(masked_logits / temperature).numpy()
        return np.random.choice(self.action_dim, p=probs)

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)

        self.reward_buffer.extend(returns)
        if len(self.reward_buffer) > 1000:
            self.reward_buffer = self.reward_buffer[-1000:]

        # Normalisation
        if len(self.reward_buffer) > 0:
            returns = (returns - np.mean(self.reward_buffer)) / (np.std(self.reward_buffer) + 1e-8)
        return returns

    def train_episode(self, env):
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False

        while not done:
            valid_actions = env.available_actions_ids()
            action = self.get_action(state, valid_actions)

            env.step(action)
            reward = env.score() if env.is_game_over() else 0
            next_state = env.state_description()
            done = env.is_game_over()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = self.compute_returns(rewards)

        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.policy(states, training=True)

            # Calcul des probabilités pour les actions choisies
            action_masks = tf.one_hot(actions, self.action_dim)
            selected_logits = tf.reduce_sum(logits * action_masks, axis=1)

            # Log probabilities et loss
            log_probs = selected_logits - tf.reduce_logsumexp(logits, axis=1)
            basic_loss = -tf.reduce_mean(log_probs * returns)

            # Ajout de l'entropie
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            loss = basic_loss - 0.01 * entropy

        # Calcul et application des gradients
        grads = tape.gradient(loss, self.policy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        return sum(rewards), loss.numpy()

    def train(self, env, episodes=5000):
        history = []
        window_size = 100

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            total_reward, loss = self.train_episode(env)
            history.append(total_reward)

            if (episode + 1) % 100 == 0:
                recent_rewards = history[-window_size:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([r > 0 for r in recent_rewards])
                print(f"Episode {episode + 1}")
                print(f"Moyenne des récompenses: {avg_reward:.2f}")
                print(f"Taux de victoire: {win_rate:.2%}")
                print(f"Loss: {loss:.6f}\n")
        self.save_model('reinforce_model_Farkel.h5')

        return history

    def save_model(self, filepath):
        self.policy.save(filepath)


if __name__ == "__main__":
    from environment.tictactoe import TicTacToe
    from environment.FarkelEnv import  FarkleDQNEnv

    # Configuration de TensorFlow pour moins de warnings
    tf.get_logger().setLevel('ERROR')

    env = TicTacToe()
    env = FarkleDQNEnv()
    agent = REINFORCE(
        state_dim=12,
        action_dim=128,
        alpha=0.0001,
        gamma=0.99
    )

    history = agent.train(env, episodes=50000)
    #model = load_model('reinforce_model_tictactoe.h5')
