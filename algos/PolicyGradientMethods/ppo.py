import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
from statistics import mean


class PPO:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, alpha=0.0003):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

        # Paramètres d'exploration
        self.epsilon = 1.0

        # Réseaux et optimiseurs
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def _build_actor(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])

    def _build_critic(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def select_action(self, state_tensor, action_mask, training=True):
        """Sélectionne une action avec masquage des actions invalides"""
        # Obtention des probabilités d'action
        probs = self.actor(state_tensor[None])[0].numpy()

        # Masquage des actions invalides
        mask = np.ones_like(probs) * float('-inf')
        valid_actions = np.where(action_mask == 1)[0]
        mask[valid_actions] = 0
        masked_probs = tf.nn.softmax(probs + mask).numpy()

        # Exploration epsilon-greedy en training
        if training and np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            action = valid_actions[np.argmax(masked_probs[valid_actions])]

        return action, masked_probs

    def train_episode(self, env, episode_num):
        states, actions, rewards, values, old_probs = [], [], [], [], []
        state = env.state_description()
        done = False
        episode_reward = 0

        # Ajustement exploration
        if episode_num < 20000:
            self.epsilon = max(0.01, 1.0 - (episode_num / 20000) * 0.5)

        while not done:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            action_mask = env.action_mask()

            # Sélection d'action
            action, action_probs = self.select_action(state_tensor, action_mask)

            # Exécution de l'action
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            next_state = env.state_description()
            done = env.is_game_over()

            # Stockage des données
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(self.critic(state_tensor[None]).numpy()[0, 0])
            old_probs.append(action_probs[action])

            state = next_state
            episode_reward += reward

        # Conversion en arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        old_probs = np.array(old_probs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        # Calcul des avantages et returns
        returns = self.compute_returns(rewards)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update des réseaux
        actor_loss = self.update_actor(states, actions, old_probs, advantages)
        critic_loss = self.update_critic(states, returns)

        return episode_reward, actor_loss, critic_loss

    def compute_returns(self, rewards):
        returns = np.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def update_actor(self, states, actions, old_probs, advantages):
        with tf.GradientTape() as tape:
            # Nouvelles probabilités
            new_probs = self.actor(states)
            actions_one_hot = tf.one_hot(actions, self.action_dim)
            new_action_probs = tf.reduce_sum(new_probs * actions_one_hot, axis=1)

            # Ratio et clipping
            ratios = new_action_probs / (old_probs + 1e-10)
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            # Pertes PPO
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        # Application des gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss.numpy()

    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return critic_loss.numpy()

    def train(self, env, episodes=50000, eval_interval=1000):
        for episode in tqdm(range(episodes), desc="Training"):
            env.reset()
            reward, actor_loss, critic_loss = self.train_episode(env, episode)

            if (episode + 1) % eval_interval == 0:
                eval_reward, eval_steps = self.evaluate(env)
                print(f"\nÉpisode {episode + 1}")
                print(f"Reward moyen: {eval_reward:.2f}")
                print(f"Steps moyen: {eval_steps:.2f}")
                print(f"Actor Loss: {actor_loss:.6f}")
                print(f"Critic Loss: {critic_loss:.6f}")
                print(f"Epsilon: {self.epsilon:.4f}")

    def evaluate(self, env, episodes=100):
        rewards = []
        steps = []

        for _ in range(episodes):
            env.reset()
            state = env.state_description()
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                action_mask = env.action_mask()
                action, _ = self.select_action(state_tensor, action_mask, training=False)

                _, reward, done, _, _ = env.step(action)
                state = env.state_description()
                episode_reward += reward
                episode_steps += 1

            rewards.append(episode_reward)
            steps.append(episode_steps)

        return np.mean(rewards), np.mean(steps)