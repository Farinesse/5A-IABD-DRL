import tensorflow as tf
import keras
import numpy as np
from tqdm import tqdm
import time
from statistics import mean
from functions.outils import log_metrics_to_dataframe, plot_csv_data


class REINFORCEBaseline:
    def __init__(self, state_dim, action_dim, alpha_theta=0.0001, alpha_w=0.0001, gamma=0.99, path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha_theta = alpha_theta  # Learning rate for policy (θ)
        self.alpha_w = alpha_w  # Learning rate for baseline (w)
        self.gamma = gamma  # discount factor
        self.path = path

        # Initialize networks
        self.policy = self._build_policy()
        self.baseline = self._build_baseline()

        # Initialize optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_theta)
        self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_w)

        self.reward_buffer = []

    def _build_policy(self):
        # π(a|s,θ) - policy parameterization
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.action_dim, activation='softmax')  # Sortie en distribution de probabilités
        ])

    def _build_baseline(self):
        # v̂(s,w) - state-value function
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1)
        ])

    def compute_returns(self, rewards):
        # Calcul de Gt selon la formule Σ(k=t+1 to T) γ^(k-t-1) * Rk
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        return returns

    def train_episode(self, env):
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False
        time_steps = []
        t = 0

        while not done:
            valid_actions = env.available_actions_ids()
            state_tensor = tf.convert_to_tensor(np.array(state).reshape(1, -1), dtype=tf.float32)

            # π(a|s,θ) - Calcul des probabilités d'action
            probs = self.policy(state_tensor, training=False)[0].numpy()

            # Masquer les actions invalides
            mask = np.ones_like(probs) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(probs + mask).numpy()

            # Sélection d'action selon π(a|s,θ)
            epsilon = max(0.01, (1 - len(self.reward_buffer) / 30000))
            if np.random.random() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                action = np.random.choice(self.action_dim, p=masked_probs)

            # Exécuter l'action et observer R, S'
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

        # Conversion en tensors
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        time_steps = np.array(time_steps, dtype=np.float32)
        returns = self.compute_returns(rewards)

        # Update baseline (state-value function)
        with tf.GradientTape() as tape:
            baseline_values = tf.squeeze(self.baseline(states))
            baseline_loss = tf.reduce_mean(tf.square(returns - baseline_values))

        baseline_grads = tape.gradient(baseline_loss, self.baseline.trainable_variables)
        self.baseline_optimizer.apply_gradients(zip(baseline_grads, self.baseline.trainable_variables))

        # Update policy
        with tf.GradientTape() as tape:
            logits = self.policy(states, training=True)
            action_masks = tf.one_hot(actions, self.action_dim)
            log_probs = tf.reduce_sum(tf.math.log(logits + 1e-10) * action_masks, axis=1)

            # Calculate advantage (δ)
            advantages = returns - tf.stop_gradient(tf.squeeze(self.baseline(states)))

            # γ^t δ
            gamma_t = tf.pow(self.gamma, time_steps)
            time_discounted_advantages = advantages * gamma_t

            # Loss function (négatif car on veut maximiser)
            loss = -tf.reduce_mean(log_probs * time_discounted_advantages)

        # Update policy parameters θ
        policy_grads = tape.gradient(loss, self.policy.trainable_variables)
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, 0.5)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        return sum(rewards), loss.numpy()

    def train(self, env, episodes=20000):
        interval = 100
        results_df = None

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            reward, loss = self.train_episode(env)
            self.reward_buffer.append(reward)

            if (episode + 1) % interval == 0 and episode > 0:
                results_df = log_metrics_to_dataframe(
                    function=play_with_reinforce_baseline,
                    model=self.policy,
                    predict_func=None,
                    env=env,
                    episode_index=episode,
                    games=1000,
                    dataframe=results_df
                )
                print(f"Loss: {loss:.6f}")

        if self.path is not None:
            self.save_models(self.path)
            results_df.to_csv(f"{self.path}_metrics.csv", index=False)

    def save_models(self, filepath):
        self.policy.save(f"policy_{filepath}")
        self.baseline.save(f"baseline_{filepath}")


def play_with_reinforce_baseline(env, model, predict_func=None, episodes=100):
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
            valid_actions = env.available_actions_ids()

            probs = model(tf.expand_dims(state_tensor, 0), training=False)[0]
            mask = np.ones_like(probs.numpy()) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(probs + mask).numpy()

            if len(valid_actions) > 0:
                action = valid_actions[np.argmax(masked_probs[valid_actions])]
            else:
                print("Aucune action valide disponible!")
                action = np.random.choice(env.available_actions_ids())

            env.step(action)
            nb_turns += 1

        end_time = time.time()
        if nb_turns == 100:
            episode_scores.append(-1)
        else:
            episode_scores.append(env.score())

        episode_time = end_time - start_time
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
    from environment.tictactoe import TicTacToe

    tf.get_logger().setLevel('ERROR')
    env = TicTacToe()

    agent = REINFORCEBaseline(
        state_dim=27,
        action_dim=9,
        alpha_theta=0.001,
        alpha_w=0.001,
        gamma=0.99,
        path='tictactoe_reinforce_baseline.h5'
    )

    agent.train(env, episodes=50000)
    plot_csv_data(agent.path + "_metrics.csv")