import csv
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from environment.FarkelEnv import FarkleDQNEnv


class PPO_A2C_Style:
    def __init__(self, state_dim, action_dim, alpha=0.0003, gamma=0.99, clip_ratio=0.2, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay_episodes=20000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        # Exploration (epsilon-greedy)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_episodes

        # Modèle combiné pour la politique (acteur) et le critique
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def _build_model(self):
        """Construit un réseau combiné pour la politique et le critique."""
        inputs = keras.layers.Input(shape=(self.state_dim,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)

        # Sortie pour la politique (distribution des actions)
        policy = keras.layers.Dense(self.action_dim)(x)
        # Sortie pour la valeur d'état (critique)
        value = keras.layers.Dense(1)(x)

        return keras.Model(inputs=inputs, outputs=[policy, value])

    def select_action(self, state, valid_actions):
        """Sélectionne une action avec exploration epsilon-greedy et masquage des actions invalides."""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        logits, _ = self.model(state_tensor)
        logits = logits.numpy()[0]

        # Masquage des actions invalides
        mask = np.ones_like(logits) * float('-inf')
        mask[valid_actions] = 0
        masked_logits = logits + mask

        # Softmax pour obtenir les probabilités des actions
        probs = tf.nn.softmax(masked_logits).numpy()

        # Exploration epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            action = valid_actions[np.argmax(probs[valid_actions])]

        return action, probs

    def compute_advantages(self, rewards, values):
        """Calcule les avantages (GAE) et retours pour PPO."""
        returns, advantages = [], []
        G = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            G = r + self.gamma * G
            returns.insert(0, G)
            advantages.insert(0, G - v)
        advantages = np.array(advantages)
        return np.array(returns), (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def train_episode(self, env):
        """Entraîne l'agent sur un épisode unique."""
        states, actions, rewards, old_probs, values = [], [], [], [], []
        state = env.state_description()
        done = False
        episode_reward = 0

        while not done:
            valid_actions = env.available_actions_ids()
            action, probs = self.select_action(state, valid_actions)

            prev_score = env.score()

            # Exécuter l'action
            env.step(action)
            #print(f"descrip {env.state_description()}, Des = {env.dice_roll}, action = {env.decode_action(action)}, etat  = {env.get_observation()}")
            reward = env.score() - prev_score
            next_state = env.state_description()
            done = env.is_game_over()

            # Enregistrement des données
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_probs.append(probs[action])
            _, value = self.model(tf.convert_to_tensor([state], dtype=tf.float32))
            values.append(value.numpy()[0, 0])

            state = next_state
            episode_reward += reward

        # Mise à jour d'epsilon (exploration)
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_rate)

        # Conversion en arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        old_probs = np.array(old_probs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        # Calcul des avantages et retours
        returns, advantages = self.compute_advantages(rewards, values)

        # Mise à jour via PPO
        loss, policy_loss, critic_loss = self.update_policy(states, actions, old_probs, returns, advantages)
        return episode_reward, loss, policy_loss, critic_loss

    def update_policy(self, states, actions, old_probs, returns, advantages):
        """Met à jour la politique via PPO avec les gradients clippés."""
        with tf.GradientTape() as tape:
            logits, values = self.model(states)
            values = tf.squeeze(values)

            # Calcul des probabilités des actions prises
            action_masks = tf.one_hot(actions, self.action_dim)
            new_probs = tf.reduce_sum(action_masks * tf.nn.softmax(logits), axis=1)

            # Ratio PPO
            ratios = new_probs / (old_probs + 1e-10)
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratios * advantages, clipped_ratios * advantages)
            )

            # Perte critique
            critic_loss = tf.reduce_mean(tf.square(returns - values))

            # Perte totale
            total_loss = policy_loss + 0.5 * critic_loss

        # Application des gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return total_loss, policy_loss, critic_loss

    import random

    def evaluate_policy(self, env: FarkleDQNEnv, episodes=100):
        """
        Évalue une politique dans l'environnement Farkle.

        Args:
            env (FarkleDQNEnv): L'environnement Farkle.
            episodes (int): Nombre d'épisodes pour l'évaluation.

        Returns:
            tuple: Moyennes des récompenses, longueurs des parties et temps d'exécution.
        """
        total_rewards = []
        total_lengths = []
        total_times = []

        for _ in range(episodes):
            env.reset()
            state = env.state_description()
            done = False
            rewards = 0
            length = 0

            while not done:
                # Tour du joueur 1 (agent)
                valid_actions = env.available_actions_ids()
                action, _ = self.select_action(state, valid_actions)  # Méthode `select_action` doit être définie.
                observation, reward, done, truncated, info = env.step(action)


                if  done:
                    rewards += env.score()


                # Mise à jour de l'état
                state = env.state_description()
                length += 1

            # Enregistrer les statistiques de l'épisode
            total_rewards.append(rewards)
            total_lengths.append(length)
            total_times.append(0.001)  # Placeholder pour le temps si non calculé

        # Calculer les moyennes
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(total_lengths)
        avg_time = np.mean(total_times)

        return avg_reward, avg_length, avg_time

    def train(self, env, episodes=5000, eval_interval=500, eval_episodes=100, csv_filename="training_results_2.csv"):
        """Entraîne l'agent sur plusieurs épisodes et enregistre les résultats."""
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Episode", "Average Reward", "Max Reward", "Min Reward", "Loss", "Policy Loss", "Critic Loss"])

            for episode in tqdm(range(episodes), desc="Training Episodes"):
                env.reset()
                reward, loss, policy_loss, critic_loss = self.train_episode(env)

                # Affichage périodique
                if (episode + 1) % eval_interval == 0:
                    avg_reward, avg_length, avg_time_per_move = self.evaluate_policy(env, eval_episodes)
                    writer.writerow([episode + 1, avg_reward, avg_reward, avg_reward, loss, policy_loss, critic_loss])
                    print(
                        f"Évaluation : Épisode {episode + 1}, Moyenne = {avg_reward:.2f}, Longueur Moyenne = {avg_length:.2f}, Loss = {loss:.2f}, Policy Loss = {policy_loss:.2f}, Critic Loss = {critic_loss:.2f}, Critic Loss = {self.epsilon:.2f} ")


if __name__ == "__main__":
    env = FarkleDQNEnv(num_players=2, target_score=2000)
    agent = PPO_A2C_Style(
        state_dim=12,
        action_dim=128,
        alpha=0.0001,
        gamma=0.99,
        clip_ratio=0.2,
        epsilon_decay_episodes=50000
    )
    agent.train(env, episodes=50000)