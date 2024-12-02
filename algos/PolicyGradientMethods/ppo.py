import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
from statistics import mean

from functions.outils import log_metrics_to_dataframe, play_with_ppo, save_files


class PPOActorCritic:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, alpha=0.0003, path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.path = path

        # Initialisation des réseaux
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # Optimiseurs avec gradient clipping
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, clipnorm=0.5)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, clipnorm=0.5)

    def _build_actor(self):
        """Construction du réseau de politique"""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])

    def _build_critic(self):
        """Construction du réseau critique"""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(1)
        ])

    def select_action(self, state_tensor, action_mask, valid_actions):
        """Sélectionne une action selon la politique"""
        # Obtenir les probabilités d'action
        probs = self.actor(state_tensor[None])[0].numpy()

        # Masquer les actions invalides
        mask = np.ones_like(probs) * float('-inf')
        mask[valid_actions] = 0
        masked_probs = tf.nn.softmax(probs + mask).numpy()

        # Renormaliser si nécessaire
        masked_probs = masked_probs / (np.sum(masked_probs) + 1e-8)

        return np.random.choice(len(masked_probs), p=masked_probs)

    def compute_returns(self, rewards):
        """Calcul des retours avec GAE"""
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns, dtype=np.float32)
        # Normalisation
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def train_episode(self, env):
        """Collection de trajectoire et entraînement"""
        states, actions, rewards = [], [], []
        old_action_probs = []
        state = env.state_description()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            action_mask = env.action_mask()
            valid_actions = env.available_actions_ids()

            # Sélection d'action
            action = self.select_action(state_tensor, action_mask, valid_actions)
            old_probs = self.actor(state_tensor[None])[0].numpy()

            # Interaction avec l'environnement
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            next_state = env.state_description()
            done = env.is_game_over()

            # Stockage
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_action_probs.append(old_probs[action])

            state = next_state
            episode_reward += reward

        # Conversion en tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_action_probs = tf.convert_to_tensor(old_action_probs, dtype=tf.float32)
        returns = self.compute_returns(rewards)

        # Plusieurs époques de mise à jour PPO
        for _ in range(3):  # Nombre d'époques PPO
            with tf.GradientTape() as tape:
                # Calcul des valeurs par le critique
                values = tf.squeeze(self.critic(states))
                critic_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))

            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                # Nouvelles probabilités d'action
                action_probs = self.actor(states)
                actions_onehot = tf.one_hot(actions, self.action_dim)
                new_action_probs = tf.reduce_sum(action_probs * actions_onehot, axis=1)

                # Ratio des probabilités pour PPO
                ratio = new_action_probs / (old_action_probs + 1e-8)

                # Calcul des avantages
                advantages = returns - tf.stop_gradient(values)
                advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

                # Les deux termes de la perte PPO
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

                # Loss finale (négatif car on veut maximiser)
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                # Ajouter un terme d'entropie pour l'exploration
                entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1))
                actor_loss = actor_loss - 0.01 * entropy

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return episode_reward, float(actor_loss), float(critic_loss)

    def train(self, env, episodes=10000, interval=1000):
        results_df = None

        for episode in tqdm(range(episodes), desc="Training"):
            env.reset()
            total_reward, actor_loss, critic_loss = self.train_episode(env)

            if (episode + 1) % interval == 0:
                results_df = log_metrics_to_dataframe(
                    function=play_with_ppo,
                    model=self.actor,  # On utilise le réseau actor pour l'évaluation
                    predict_func=None,
                    env=env,
                    episode_index=episode,
                    games=100,
                    dataframe=results_df
                )
                print(f"\n{'=' * 50}")
                print(f"Episode {episode + 1}")
                print(f"Actor Loss: {actor_loss:.6f}")
                print(f"Critic Loss: {critic_loss:.6f}")

        # Utiliser save_files à la fin de l'entraînement
        if self.path is not None:
            save_files(
                online_model=self.actor,
                algo_name="PPO_ACTOR_CRITIC",
                results_df=results_df,
                env=env,
                num_episodes=episodes,
                gamma=self.gamma,
                alpha=self.clip_epsilon,
                optimizer=self.actor_optimizer,
                save_path=self.path
            )

        return results_df

    def evaluate(self, env, n_episodes=100):
        """Évaluation de la politique"""
        rewards = []
        for _ in range(n_episodes):
            env.reset()
            done = False
            total_reward = 0
            state = env.state_description()

            while not done:
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                action_mask = env.action_mask()
                valid_actions = env.available_actions_ids()
                action = self.select_action(state_tensor, action_mask, valid_actions)

                env.step(action)
                reward = env.score()
                done = env.is_game_over()
                state = env.state_description()
                total_reward += reward

            rewards.append(total_reward)

        return rewards


if __name__ == "__main__":
    from environment.line_word import LineWorld

    env = LineWorld(10)
    agent = PPOActorCritic(
        state_dim=10,
        action_dim=3,
        clip_epsilon=0.2,
        gamma=0.99,
        alpha=0.0003,
        path='ppo_actor_critic'
    )

    agent.train(env, episodes=200)