import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, alpha_theta=0.001, alpha_w=0.001, gamma=0.99):
        # Initialize policy parameter θ ∈ ℝᵈ and state-value weights w ∈ ℝᵈ
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha_theta = alpha_theta  # Step size αθ
        self.alpha_w = alpha_w  # Step size αw
        self.gamma = gamma

        # π(a|s,θ) - policy parameterization
        self.policy = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.action_dim, activation='softmax')
        ])

        # v̂(s,w) - state-value function parameterization
        self.baseline = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1)  # Estimation de la valeur d'état
        ])

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_theta)
        self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_w)

    def compute_G(self, rewards, t):
        """Compute Gt = Σᵏ₌ₜ₊₁ᵀ γᵏ⁻ᵗ⁻¹Rₖ"""
        G = 0
        for k in range(t + 1, len(rewards)):
            G += (self.gamma ** (k - t - 1)) * rewards[k]
        return G

    def train_episode(self, env):
        # Generate an episode S₀,A₀,R₁,...,Sₜ₋₁,Aₜ₋₁,Rₜ, following π(·|·,θ)
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False

        while not done:
            valid_actions = env.available_actions_ids()

            # Get action probabilities from policy π(·|s,θ)
            state_tensor = tf.convert_to_tensor(np.array(state).reshape(1, -1), dtype=tf.float32)
            probs = self.policy(state_tensor, training=False)[0].numpy()

            # Mask invalid actions
            mask = np.ones_like(probs) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(probs + mask).numpy()

            # Sample action from policy
            action = np.random.choice(self.action_dim, p=masked_probs)

            # Execute action
            env.step(action)
            reward = env.score() if env.is_game_over() else 0
            next_state = env.state_description()
            done = env.is_game_over()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Loop for each step of the episode t = 0,1,...,T-1:
        total_loss = 0
        for t in range(len(states)):
            state_t = np.array(states[t], dtype=np.float32).reshape(1, -1)
            action_t = actions[t]

            # G ← Σᵏ₌ₜ₊₁ᵀ γᵏ⁻ᵗ⁻¹Rₖ
            G = self.compute_G(rewards, t)

            # δ ← G - v̂(Sₜ,w)
            with tf.GradientTape() as tape:
                baseline_value = self.baseline(state_t)
                delta = G - tf.squeeze(baseline_value)
                # w ← w + αᵂδ∇v̂(Sₜ,w)
                baseline_loss = tf.square(delta)

            baseline_grads = tape.gradient(baseline_loss, self.baseline.trainable_variables)
            self.baseline_optimizer.apply_gradients(zip(baseline_grads, self.baseline.trainable_variables))

            # θ ← θ + αθγᵗδ∇ln π(Aₜ|Sₜ,θ)
            with tf.GradientTape() as tape:
                logits = self.policy(state_t)
                action_mask = tf.one_hot([action_t], self.action_dim)
                log_prob = tf.reduce_sum(tf.math.log(logits + 1e-10) * action_mask)
                policy_loss = -(self.gamma ** t) * delta * log_prob

            policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

            total_loss += policy_loss.numpy()

        return sum(rewards), total_loss

    def train(self, env, episodes=10000):
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

        self.save_models('baseline_policy.h5', 'baseline_value.h5')
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

    history = agent.train(env, episodes=10000)