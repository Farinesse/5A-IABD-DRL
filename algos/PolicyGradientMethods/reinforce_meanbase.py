import tensorflow as tf
import keras
import numpy as np
from tqdm import tqdm
import time
from statistics import mean

from environment.line_word import LineWorld
from functions.outils import log_metrics_to_dataframe, plot_csv_data, save_files


class REINFORCEMeanBaseline:
    def __init__(self, state_dim, action_dim, alpha=0.0001, gamma=0.99, path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha  # Learning rate for policy
        self.gamma = gamma
        self.path = path

        # Initialize policy network
        self.policy = self._build_policy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

        # Pour le suivi des performances
        self.reward_buffer = []

    def _build_policy(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer='glorot_normal',
                                  bias_initializer='zeros'),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer='glorot_normal',
                                  bias_initializer='zeros'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax',
                                  kernel_initializer='glorot_normal',
                                  bias_initializer='zeros')
        ])

    def compute_returns(self, rewards):
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

            # Sélection d'action
            action = np.random.choice(self.action_dim, p=masked_probs)

            # Exécuter l'action
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
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        time_steps = tf.convert_to_tensor(time_steps, dtype=tf.float32)
        returns = self.compute_returns(rewards)

        # Calcul du mean baseline
        baseline = tf.reduce_mean(returns)

        # Update policy
        with tf.GradientTape() as tape:
            logits = self.policy(states)
            action_masks = tf.one_hot(actions, self.action_dim)
            log_probs = tf.reduce_sum(tf.math.log(logits + 1e-10) * action_masks, axis=1)

            # Avantages avec mean baseline
            advantages = returns - baseline

            # Pondération temporelle
            gamma_t = tf.pow(self.gamma, time_steps)
            time_discounted_advantages = advantages * gamma_t

            # Loss function
            loss = -tf.reduce_mean(log_probs * time_discounted_advantages)

        # Update policy parameters
        grads = tape.gradient(loss, self.policy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        return sum(rewards), float(loss)

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
                    games=100,
                    dataframe=results_df
                )
                print(f"Loss: {loss:.6f}")

        if self.path is not None:
            save_files(
                online_model=self.policy,
                algo_name="REINFORCE_MEAN_BASELINE",
                results_df=results_df,
                env=env,
                num_episodes=episodes,
                gamma=self.gamma,
                alpha=self.alpha,
                optimizer=self.optimizer,
                save_path=self.path
            )

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
            episode_scores.append(env.score(testing=True))

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
    env = LineWorld(10)

    agent = REINFORCEMeanBaseline(
        state_dim=10,
        action_dim=3,
        alpha=0.001,
        gamma=0.99,
        path='tictactoe_reinforce_baseline'
    )

    agent.train(env, episodes=200)
    plot_csv_data(agent.path + "_metrics.csv")
