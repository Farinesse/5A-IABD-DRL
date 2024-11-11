
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class REINFORCEWithCritic:
    def __init__(self, state_dim, action_dim, alpha_policy=0.001, alpha_critic=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Construction des réseaux
        self.policy = self._build_policy()
        self.critic = self._build_critic()

        # Optimiseurs séparés pour la politique et le critique
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_policy)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_critic)

    def _build_policy(self):
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.action_dim)
        ])

    def _build_critic(self):
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1)  # Estimation de V(s)
        ])

    def select_action(self, state, valid_actions):
        # Epsilon-greedy pour l'exploration
        if np.random.random() < 0.05:
            return np.random.choice(valid_actions)

        # Forward pass pour obtenir les logits
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.policy(state_tensor, training=False)[0].numpy()

        # Masquer les actions invalides
        mask = np.ones_like(logits) * float('-inf')
        mask[valid_actions] = 0
        masked_logits = logits + mask

        # Softmax avec température adaptative
        temperature = max(0.5, 1.0)  # Température fixe ou adaptative
        probs = tf.nn.softmax(masked_logits / temperature).numpy()


        return np.random.choice(self.action_dim, p=probs)

    def compute_returns(self, rewards):
        # Calcul des retours Monte Carlo
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return np.array(returns, dtype=np.float32)

    def train_episode(self, env):
        # Collecte de trajectoire
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False

        while not done:
            # Sélection et exécution de l'action
            valid_actions = env.available_actions_ids()
            action = self.select_action(state, valid_actions)

            # Exécution de l'action
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            next_state = env.state_description()
            done = env.is_game_over()

            # Stockage des transitions
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Conversion en tensors
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        state_tensors = tf.convert_to_tensor(states)

        # Calcul des retours Monte Carlo
        returns = self.compute_returns(rewards)

        # Mise à jour du critique (baseline)
        with tf.GradientTape() as critic_tape:
            # Prédiction des valeurs d'état
            values = self.critic(state_tensors)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))

        # Application des gradients du critique
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Calcul des avantages (retours - baseline)
        advantages = returns - tf.squeeze(values)

        # Mise à jour de la politique
        with tf.GradientTape() as policy_tape:
            # Forward pass de la politique
            logits = self.policy(state_tensors)

            # Calcul des probabilités des actions prises
            action_masks = tf.one_hot(actions, self.action_dim)
            selected_logits = tf.reduce_sum(logits * action_masks, axis=1)

            # Log probabilities
            log_probs = selected_logits - tf.reduce_logsumexp(logits, axis=1)

            # Loss de la politique avec avantages
            policy_loss = -tf.reduce_mean(log_probs * advantages)

            # Ajout d'un terme d'entropie pour encourager l'exploration
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            policy_loss = policy_loss - 0.01 * entropy

        # Application des gradients de la politique
        policy_grads = policy_tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        return sum(rewards), policy_loss.numpy(), critic_loss.numpy()

    def train(self, env, episodes=5000):
        history = []
        window_size = 100

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            total_reward, policy_loss, critic_loss = self.train_episode(env)
            history.append(total_reward)

            if (episode + 1) % 100 == 0:
                recent_rewards = history[-window_size:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([r > 0 for r in recent_rewards])
                print(f"\nEpisode {episode + 1}")
                print(f"Moyenne des récompenses: {avg_reward:.2f}")
                print(f"Taux de victoire: {win_rate:.2%}")
                print(f"Policy Loss: {policy_loss:.6f}")
                print(f"Critic Loss: {critic_loss:.6f}")

        self.save_models('reinforce_policy.h5', 'reinforce_critic.h5')
        return history

    def save_models(self, policy_path, critic_path):
        self.policy.save(policy_path)
        self.critic.save(critic_path)


if __name__ == "__main__":
    from environment.tictactoe import TicTacToe

    tf.get_logger().setLevel('ERROR')

    env = TicTacToe()
    agent = REINFORCEWithCritic(
        state_dim=27,
        action_dim=9,
        alpha_policy=0.001,
        alpha_critic=0.001,
        gamma=0.99
    )

    history = agent.train(env, episodes=5000)