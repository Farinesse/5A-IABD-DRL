import time
from statistics import mean
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
#from functions.outils import log_metrics_to_dataframe, plot_csv_data


class OneStepActorCritic:
    def __init__(self, state_dim, action_dim, alpha_theta=0.001, alpha_w=0.001, gamma=0.99, path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        self.path = path

        # Paramètres d'exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.temperature = 1.0
        self.noise_std = 0.1

        # Initialize networks
        self.policy = self._build_policy()
        self.value = self._build_value()

    def _build_policy(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])

    def _build_value(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def select_action(self, state_tensor, action_mask, exploration=True):
        """Sélection d'action avec exploration et masque d'action"""
        valid_actions = np.where(action_mask == 1)[0]

        # Exploration epsilon-greedy
        if exploration and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        # Exploitation avec température et bruit
        probs = self.policy(state_tensor[None])[0].numpy()

        if exploration:
            probs = probs / self.temperature
            masked_probs = probs * action_mask
            noise = np.random.normal(0, self.noise_std, size=masked_probs.shape)
            masked_probs = masked_probs + noise * action_mask
        else:
            masked_probs = probs * action_mask

        # Normalisation sûre
        masked_probs = np.clip(masked_probs, 1e-8, None)
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs = np.zeros_like(probs)
            masked_probs[valid_actions] = 1.0 / len(valid_actions)

        if not exploration:
            return valid_actions[np.argmax(masked_probs[valid_actions])]

        return np.random.choice(self.action_dim, p=masked_probs)

    def train_episode(self, env, episode_num):
        # Initialisation
        s = env.state_description()
        s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
        I = 1.0
        done = False
        total_reward = 0
        episode_steps = []

        # Ajustement exploration
        if episode_num < 50000:
            self.epsilon = max(self.epsilon_min, 1.0 - (episode_num / 50000) * 0.5)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        while not done:
            action_mask = env.action_mask()
            action = self.select_action(s_tensor, action_mask, exploration=True)

            # Exécution de l'action
            prev_score = env.score()
            env.step(action)
            next_state = env.state_description()
            reward = env.score() - prev_score
            done = env.is_game_over()

            next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Calcul TD error
                current_value = self.value(s_tensor[None])
                next_value = tf.zeros_like(current_value) if done else self.value(next_state_tensor[None])
                td_error = tf.clip_by_value(reward + self.gamma * next_value - current_value, -1.0, 1.0)

                # Value loss
                value_loss = 0.5 * tf.square(td_error)

                # Policy loss avec stabilité numérique
                action_probs = self.policy(s_tensor[None])
                action_mask_tensor = tf.convert_to_tensor(action_mask[None], dtype=tf.float32)
                masked_probs = action_probs * action_mask_tensor
                masked_probs = tf.clip_by_value(masked_probs, 1e-8, 1.0)
                normalized_probs = masked_probs / (tf.reduce_sum(masked_probs, axis=1, keepdims=True) + 1e-8)

                action_one_hot = tf.one_hot(action, self.action_dim)
                log_prob = tf.math.log(tf.reduce_sum(normalized_probs * action_one_hot, axis=1) + 1e-8)
                policy_loss = -log_prob * tf.stop_gradient(td_error) * I

            # Mise à jour critic
            value_grads = tape.gradient(value_loss, self.value.trainable_variables)
            value_grads = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else grad for grad in value_grads]

            for var, grad in zip(self.value.trainable_variables, value_grads):
                if grad is not None:
                    var.assign_add(self.alpha_w * grad)

            # Mise à jour actor
            policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
            policy_grads = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else grad for grad in policy_grads]

            for var, grad in zip(self.policy.trainable_variables, policy_grads):
                if grad is not None:
                    var.assign_add(self.alpha_theta * grad)

            del tape

            # Mise à jour variables
            I *= self.gamma
            s_tensor = next_state_tensor
            total_reward += reward

            if not (tf.math.is_nan(value_loss) or tf.math.is_nan(policy_loss)):
                episode_steps.append({
                    'value_loss': float(value_loss),
                    'policy_loss': float(policy_loss),
                    'td_error': float(td_error)
                })

        if not episode_steps:
            return total_reward, 0.0, 0.0

        return total_reward, np.mean([s['policy_loss'] for s in episode_steps]), np.mean(
            [s['value_loss'] for s in episode_steps])

    def train(self, env, episodes=100000):
        interval = 1000  # Intervalle d'évaluation
        results_df = None

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            total_reward, policy_loss, value_loss = self.train_episode(env, episode)

            if (episode + 1) % interval == 0 and episode > 0:
                results_df = log_metrics_to_dataframe(
                    function=play_with_actor_critic,
                    model=self.policy,
                    predict_func=None,
                    env=env,
                    episode_index=episode,
                    games=1000,
                    value_model=self.value,
                    dataframe=results_df
                )

                print(f"\n{'=' * 50}")
                print(f"Episode {episode + 1}")
                print(f"Policy Loss: {policy_loss:.6f}")
                print(f"Value Loss: {value_loss:.6f}")
                print(f"Epsilon: {self.epsilon:.4f}")

        if self.path is not None:
            self.save_models()
            results_df.to_csv(f"{self.path}_metrics.csv", index=False)

        return results_df

    def save_models(self):
        if self.path is not None:
            self.policy.save(f"{self.path}_policy.h5")
            self.value.save(f"{self.path}_value.h5")


def play_with_actor_critic(env, model, value_model=None, episodes=100):
    """Fonction d'évaluation"""
    episode_scores = []
    episode_times = []
    episode_steps = []
    step_times = []
    total_time = 0

    for episode in range(episodes):
        env.reset()
        nb_turns = 0

        start_time = time.time()
        while not env.is_game_over() and nb_turns < 100:
            state = env.state_description()
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            action_mask = env.action_mask()
            valid_actions = env.available_actions_ids()

            # Mode évaluation
            probs = model(tf.expand_dims(state_tensor, 0), training=False)[0].numpy()
            mask = np.ones_like(probs) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(probs + mask).numpy()

            # Action la plus probable parmi les valides
            action = valid_actions[np.argmax(masked_probs[valid_actions])] if len(
                valid_actions) > 0 else np.random.choice(valid_actions)

            env.step(action)
            nb_turns += 1

        end_time = time.time()
        episode_time = end_time - start_time

        episode_scores.append(env.score() if nb_turns < 100 else -1)
        episode_times.append(episode_time)
        total_time += episode_time
        episode_steps.append(nb_turns)
        step_times.append(episode_time / nb_turns)

    return (
        mean(episode_scores),
        mean(episode_times),
        mean(episode_steps),
        mean(step_times),
        episode_scores.count(1.0) / episodes
    )


if __name__ == "__main__":
    from environment.FarkelEnv import FarkleDQNEnv

    env = FarkleDQNEnv(target_score=5000)
    agent = OneStepActorCritic(
        state_dim=12,
        action_dim=128,
        alpha_theta=0.0001,
        alpha_w=0.0001,
        gamma=0.99,
        path='actor_critic_model'
    )

    print("Starting training...")
    results_df = agent.train(env, episodes=100000)
    plot_csv_data(agent.path + "_metrics.csv")
    print("\nTraining completed!")
