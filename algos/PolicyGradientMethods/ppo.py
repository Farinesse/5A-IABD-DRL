import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class PPO_A2C_Style:
    def __init__(self, state_dim, action_dim, alpha=0.0003, gamma=0.99, clip_ratio=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        # Modèle combiné pour la politique et le critique
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def _build_model(self):
        inputs = keras.layers.Input(shape=(self.state_dim,))
        common = keras.layers.Dense(128, activation='relu')(inputs)
        common = keras.layers.Dense(256, activation='relu')(common)

        # Politique (acteur)
        policy = keras.layers.Dense(self.action_dim)(common)

        # Critique (valeur d'état)
        value = keras.layers.Dense(1)(common)

        return keras.Model(inputs=inputs, outputs=[policy, value])

    def select_action(self, state, valid_actions):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        logits, _ = self.model(state_tensor)
        logits = logits.numpy()[0]

        # Masquer les actions invalides
        mask = np.ones_like(logits) * float('-inf')
        mask[valid_actions] = 0
        masked_logits = logits + mask

        # Softmax pour obtenir les probabilités
        probs = tf.nn.softmax(masked_logits).numpy()

        # Epsilon-greedy
        if np.random.random() < 0.05:
            action = np.random.choice(valid_actions)
        else:
            # Sélectionner une action parmi les actions valides
            valid_probs = probs[valid_actions]
            valid_probs = valid_probs / np.sum(valid_probs)
            action = valid_actions[np.random.choice(len(valid_actions), p=valid_probs)]

        return action, probs

    def compute_advantages(self, rewards, values):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Normalisation des avantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def train_episode(self, env):
        states, actions, rewards, old_probs, values = [], [], [], [], []
        state = env.state_description()
        done = False
        episode_reward = 0

        while not done:
            valid_actions = env.available_actions_ids()
            action, probs = self.select_action(state, valid_actions)

            # Vérifier si l'action est valide
            if action not in valid_actions:
                action = np.random.choice(valid_actions)

            # Exécuter l'action
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            next_state = env.state_description()
            done = env.is_game_over()

            # Enregistrer les données
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_probs.append(probs[action])

            # Prédire la valeur d'état
            _, value = self.model(tf.convert_to_tensor([state], dtype=tf.float32))
            values.append(value.numpy()[0, 0])

            state = next_state
            episode_reward += reward

        # Conversion en arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        old_probs = np.array(old_probs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        # Calcul des avantages et retours
        returns, advantages = self.compute_advantages(rewards, values)

        # Multiple epochs d'optimisation PPO
        for _ in range(4):
            self.update_policy(states, actions, old_probs, returns, advantages)

        return episode_reward

    def update_policy(self, states, actions, old_probs, returns, advantages):
        with tf.GradientTape() as tape:
            # Forward pass
            probs, values = self.model(states)
            values = tf.squeeze(values)

            # Masque pour les actions prises
            action_masks = tf.one_hot(actions, self.action_dim)
            new_probs = tf.reduce_sum(action_masks * tf.nn.softmax(probs), axis=1)

            # Ratio PPO
            ratios = new_probs / (old_probs + 1e-10)

            # Pertes clippées et non clippées
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(
                tf.minimum(
                    ratios * advantages,
                    clipped_ratios * advantages
                )
            )

            # Perte du critique
            critic_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))

            # Entropie pour l'exploration
            entropy = -tf.reduce_mean(
                tf.reduce_sum(tf.nn.softmax(probs) * tf.nn.log_softmax(probs), axis=1)
            )

            # Perte totale
            total_loss = policy_loss + critic_loss - 0.01 * entropy

        # Calcul et application des gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self, env, episodes=5000):
        history = []
        window_size = 100

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            episode_reward = self.train_episode(env)
            history.append(episode_reward)

            if (episode + 1) % 100 == 0:
                recent_rewards = history[-window_size:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([r > 0 for r in recent_rewards])
                print(f"\nEpisode {episode + 1}")
                print(f"Moyenne des récompenses: {avg_reward:.2f}")
                print(f"Taux de victoire: {win_rate:.2%}")

        self.save_model('ppo_a2c_model.h5')
        return history

    def save_model(self, filepath):
        self.model.save(filepath)


if __name__ == "__main__":
    from environment.tictactoe import TicTacToe

    tf.get_logger().setLevel('ERROR')

    env = TicTacToe()
    agent = PPO_A2C_Style(
        state_dim=27,
        action_dim=9,
        alpha=0.0003,
        gamma=0.99,
        clip_ratio=0.2
    )

    history = agent.train(env, episodes=50000)
