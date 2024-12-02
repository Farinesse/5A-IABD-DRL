import time
from statistics import mean

import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from environment.line_word import LineWorld
from functions.outils import log_metrics_to_dataframe, plot_csv_data, play_with_reinforce, save_files


class REINFORCE:
    def __init__(self, state_dim, action_dim, alpha=0.0001, gamma=0.99, path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha  # step size α
        self.gamma = gamma  # discount factor
        self.policy = self._build_policy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.reward_buffer = []
        self.path = path

    def _build_policy(self):
        # π(a|s,θ) - policy parameterization
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_dim, activation='softmax')  # Sortie en distribution de probabilités
        ])

    def select_action(self, state, valid_actions):
        """
        Sélectionne une action selon la politique courante avec exploration epsilon-greedy
        Args:
            state: État courant
            valid_actions: Liste des actions valides
        Returns:
            L'action sélectionnée
        """
        state_tensor = tf.convert_to_tensor(np.array(state).reshape(1, -1), dtype=tf.float32)

        # π(a|s,θ) - Calcul des probabilités d'action
        probs = self.policy(state_tensor, training=False)[0].numpy()

        # Masquer les actions invalides
        mask = np.ones_like(probs) * float('-inf')
        mask[valid_actions] = 0
        masked_probs = tf.nn.softmax(probs + mask).numpy()

        return np.random.choice(self.action_dim, p=masked_probs)

    def compute_returns(self, rewards):
        # Calcul de Gt selon la formule Σ(k=t+1 to T) γ^(k-t-1) * Rk
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)

        """# Normalisation des retours pour la stabilité
        self.reward_buffer.extend(returns)
        if len(self.reward_buffer) > 2000:  # Buffer plus grand
            self.reward_buffer = self.reward_buffer[-2000:]

        if len(self.reward_buffer) > 0:
            returns = (returns - np.mean(self.reward_buffer)) / (np.std(self.reward_buffer) + 1e-8)
        """
        return returns


    def train_episode(self, env):
        # Generate an episode S0,A0,R1,...,ST−1,AT−1,RT
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False
        time_steps = []
        t = 0

        while not done:
            valid_actions = env.available_actions_ids()
            state_tensor = tf.convert_to_tensor(np.array(state).reshape(1, -1), dtype=tf.float32)

            action = self.select_action(state, valid_actions)

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

        # θ ← θ + αγ^t G∇ln π(At|St,θ)
        with tf.GradientTape() as tape:
            logits = self.policy(states, training=True)

            # ∇ln π(At|St,θ)
            action_masks = tf.one_hot(actions, self.action_dim)
            log_probs = tf.reduce_sum(tf.math.log(logits + 1e-10) * action_masks, axis=1)

            # γ^t G
            gamma_t = tf.pow(self.gamma, time_steps)
            time_discounted_returns = returns * gamma_t

            # Loss function (négatif car on veut maximiser)
            loss = -tf.reduce_mean(log_probs * time_discounted_returns)

        # Mise à jour des paramètres θ
        grads = tape.gradient(loss, self.policy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)  # Pour la stabilité
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        return sum(rewards), loss.numpy()

    def train(self, env, episodes=20000):
        interval = 1000
        results_df = None

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            _, loss = self.train_episode(env)

            if (episode + 1) % interval == 0 and episode > 0:

                results_df = log_metrics_to_dataframe(
                    function = play_with_reinforce,
                    model = self.policy,
                    predict_func = None,
                    env = env,
                    episode_index = episode,
                    games = 1000,
                    dataframe = results_df
                )
                print(f"Loss: {loss:.6f}")

        if self.path is not None:
            save_files(
                online_model=self.policy,
                algo_name="REINFORCE",
                results_df=results_df,
                env=env,
                num_episodes=episodes,
                gamma=self.gamma,
                alpha=self.alpha,
                optimizer=self.optimizer,
                save_path=self.path
            )
            results_df.to_csv(f"{self.path}_metrics.csv", index=False)

    def save_model(self, filepath):
        """Sauvegarde complète du modèle et des hyperparamètres"""
        model_info = {
            'model': self.policy,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'alpha': self.alpha,
                'gamma': self.gamma
            },
            'optimizer_config': self.optimizer.get_config()
        }




if __name__ == "__main__":
    from environment.tictactoe import TicTacToe

    tf.get_logger().setLevel('ERROR')

    env = TicTacToe()
    #env = LineWorld()

    agent = REINFORCE(
        state_dim=27,
        action_dim=9,
        alpha=0.0001,
        gamma=0.99,
        path='tictactoe_reinforce_model'
    )

    agent.train(env, episodes=100000)
    #plot_csv_data(agent.path + "_metrics.csv")