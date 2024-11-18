import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, alpha_theta=0.001, alpha_w=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha_theta = alpha_theta  # Learning rate pour la politique (θ)
        self.alpha_w = alpha_w  # Learning rate pour la baseline (w)
        self.gamma = gamma

        # Construction des réseaux
        self.policy = self._build_policy()  # π(a|s,θ)
        self.baseline = self._build_baseline()  # v̂(s,w)

        # Optimiseurs séparés pour politique et baseline
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_theta)
        self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_w)

    def _build_policy(self):
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.action_dim)
        ])

    def _build_baseline(self):
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1)  # Estimation de la valeur d'état
        ])

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return np.array(returns, dtype=np.float32)

    def train_episode(self, env):
        states, actions, rewards = [], [], []
        state = env.state_description()
        time_steps = []
        t = 0
        done = False

        while not done:
            valid_actions = env.available_actions_ids()
            # Forward pass pour la politique
            logits = self.policy(np.array(state).reshape(1, -1), training=False)
            logits = logits.numpy()[0]

            # Masquer les actions invalides
            mask = np.ones_like(logits) * float('-inf')
            mask[valid_actions] = 0
            masked_logits = logits + mask

            # Sélection d'action
            probs = tf.nn.softmax(masked_logits).numpy()
            if np.random.random() < 0.05:
                action = np.random.choice(valid_actions)
            else:
                action = np.random.choice(self.action_dim, p=probs)

            # Exécution de l'action
            env.step(action)
            reward = env.score() if env.is_game_over() else 0
            next_state = env.state_description()
            done = env.is_game_over()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            time_steps.append(t)

            state = next_state
            t += 1

        # Conversion en arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        time_steps = np.array(time_steps, dtype=np.float32)
        returns = self.compute_returns(rewards)

        # Mise à jour de la baseline (critique)
        with tf.GradientTape() as tape:
            baseline_values = tf.squeeze(self.baseline(states))
            # MSE entre retours et valeurs estimées
            baseline_loss = tf.reduce_mean(tf.square(returns - baseline_values))

        baseline_grads = tape.gradient(baseline_loss, self.baseline.trainable_variables)
        self.baseline_optimizer.apply_gradients(zip(baseline_grads, self.baseline.trainable_variables))

        # Calcul des avantages (δ dans l'algorithme)
        advantages = returns - tf.squeeze(self.baseline(states))

        # Mise à jour de la politique
        with tf.GradientTape() as tape:
            logits = self.policy(states, training=True)
            action_masks = tf.one_hot(actions, self.action_dim)
            selected_logits = tf.reduce_sum(logits * action_masks, axis=1)
            log_probs = selected_logits - tf.reduce_logsumexp(logits, axis=1)

            # Application de γᵗδ
            gamma_t = tf.pow(self.gamma, time_steps)
            policy_loss = -tf.reduce_mean(log_probs * advantages * gamma_t)

        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, 0.5)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        return sum(rewards), policy_loss.numpy(), baseline_loss.numpy()

    def train(self, env, episodes=5000):
        history = []
        window_size = 100

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            total_reward, policy_loss, baseline_loss = self.train_episode(env)
            history.append(total_reward)

            if (episode + 1) % 100 == 0:
                recent_rewards = history[-window_size:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([r > 0 for r in recent_rewards])
                print(f"Episode {episode + 1}")
                print(f"Moyenne des récompenses: {avg_reward:.2f}")
                print(f"Taux de victoire: {win_rate:.2%}")
                print(f"Policy Loss: {policy_loss:.6f}")
                print(f"Baseline Loss: {baseline_loss:.6f}\n")

        self.save_models('reinforce_baseline_policy_tictactoe.h5', 'reinforce_baseline_value.h5')
        return history

    def save_models(self, policy_path, baseline_path):
        self.policy.save(policy_path)
        self.baseline.save(baseline_path)


if __name__ == "__main__":
    from environment.tictactoe import TicTacToe

    tf.get_logger().setLevel('ERROR')

    env = TicTacToe()
    agent = REINFORCEWithBaseline(
        state_dim=27,
        action_dim=9,
        alpha_theta=0.001,
        alpha_w=0.001,
        gamma=0.99
    )

    history = agent.train(env, episodes=5000)
