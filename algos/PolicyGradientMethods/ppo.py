import csv
import time
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from environment.FarkelEnv import FarkleDQNEnv
from environment.tictactoe import TicTacToe
import pandas as pd
import matplotlib.pyplot as plt


class PPO_A2C_Style:
    def __init__(self, state_dim, action_dim, n_actors=8, timesteps=128, epochs=10,
                 batch_size=32, alpha=0.0003, gamma=0.99, clip_ratio=0.2,
                 c1=1.0, c2=0.01, lambda_gae=0.95):
        """Initialize PPO agent following the paper's architecture"""
        self.state_dim = state_dim
        self.action_dim = action_dim

        # PPO parameters from the paper
        self.n_actors = n_actors  # N actors running in parallel
        self.timesteps = timesteps  # T timesteps
        self.epochs = epochs  # K epochs
        self.batch_size = batch_size  # Minibatch size M ≤ NT
        self.gamma = gamma  # Discount factor
        self.clip_ratio = clip_ratio  # ε clip ratio
        self.lambda_gae = lambda_gae  # λ for GAE
        self.c1 = c1  # c1 value function coefficient
        self.c2 = c2  # c2 entropy coefficient

        # Build policy and old policy networks
        self.policy = self._build_model()
        self.old_policy = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

        # Initialize old policy
        self.update_old_policy()

    def _build_model(self):
        """Build actor-critic network architecture"""
        inputs = keras.layers.Input(shape=(self.state_dim,))

        # Shared network
        x = keras.layers.Dense(512, activation='relu')(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.LayerNormalization()(x)

        # Policy head (actor)
        policy = keras.layers.Dense(self.action_dim, activation='softmax', name='policy')(x)

        # Value head (critic)
        value = keras.layers.Dense(1, name='value')(x)

        return keras.Model(inputs=inputs, outputs=[policy, value])

    def select_action(self, state, valid_actions, test=False):
        """Select action using the policy network"""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

        # Get policy probabilities
        policy_probs, _ = self.policy(state_tensor)

        # Mask invalid actions
        mask = np.ones_like(policy_probs.numpy()[0]) * float('-inf')
        mask[valid_actions] = 0
        masked_probs = tf.nn.softmax(policy_probs.numpy()[0] + mask).numpy()

        if test:
            # En mode test, on prend l'action avec la plus haute probabilité
            action = valid_actions[np.argmax(masked_probs[valid_actions])]
        else:
            # En mode entraînement, on échantillonne selon les probabilités
            action = np.random.choice(valid_actions,
                                      p=masked_probs[valid_actions] / np.sum(masked_probs[valid_actions]))

        return action, masked_probs
    def update_old_policy(self):
        """θold ← θ"""
        self.old_policy.set_weights(self.policy.get_weights())

    def run_actor(self, env):
        """Run policy πθ,a in environment for T timesteps"""
        states, actions, rewards, values, dones = [], [], [], [], []
        old_log_probs = []

        # Reset initial state
        env.reset()
        state = env.state_description()

        for _ in range(self.timesteps):
            valid_actions = env.available_actions_ids()

            # Si plus d'actions valides, on reset l'environnement
            if len(valid_actions) == 0:
                env.reset()
                state = env.state_description()
                valid_actions = env.available_actions_ids()

            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

            # Get policy probabilities and value
            policy_probs, value = self.policy(state_tensor)
            policy_probs = policy_probs.numpy()[0]

            # Mask invalid actions
            mask = np.ones_like(policy_probs) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(policy_probs + mask).numpy()

            # Select action
            action = np.random.choice(valid_actions,
                                      p=masked_probs[valid_actions] / np.sum(masked_probs[valid_actions]))

            # Get old policy probability
            old_policy_probs, _ = self.old_policy(state_tensor)
            old_masked_probs = tf.nn.softmax(old_policy_probs.numpy()[0] + mask).numpy()
            old_log_prob = np.log(old_masked_probs[action] + 1e-10)

            # Execute action
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            next_state = env.state_description()
            done = env.is_game_over()

            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.numpy()[0, 0])
            dones.append(done)
            old_log_probs.append(old_log_prob)

            if done:
                env.reset()
                state = env.state_description()
            else:
                state = next_state

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'dones': np.array(dones),
            'old_log_probs': np.array(old_log_probs)
        }
    def compute_gae(self, rewards, values, next_value, dones):
        """Compute GAE advantages Ât = δt + (γλ)δt+1 + ... + (γλ)^(T-t+1)δT-1"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            # δt = rt + γV(st+1) - V(st)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            # At = δt + (γλ)At+1
            advantages[t] = last_advantage = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * last_advantage

        # Returns = Advantages + Values
        returns = advantages + values
        return returns, advantages

    def optimize_surrogate(self, states, actions, old_log_probs, returns, advantages):
        """Optimize surrogate L wrt θ"""
        with tf.GradientTape() as tape:
            # Get current policy and value predictions
            policy_probs, values = self.policy(states)
            values = tf.squeeze(values)

            # Get log probabilities of actions
            action_masks = tf.one_hot(actions, self.action_dim)
            new_log_probs = tf.reduce_sum(action_masks * tf.math.log(policy_probs + 1e-10), axis=1)

            # Calculate ratio rt(θ) = πθ(at|st) / πθold(at|st)
            ratio = tf.exp(new_log_probs - old_log_probs)

            # Compute PPO-Clip objective
            # Lt^CLIP(θ) = min(rt(θ)Ât, clip(rt(θ), 1-ε, 1+ε)Ât)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )

            # Value function loss
            value_loss = tf.reduce_mean(tf.square(returns - values))

            # Entropy bonus
            entropy = -tf.reduce_mean(tf.reduce_sum(policy_probs * tf.math.log(policy_probs + 1e-10), axis=1))

            # Total loss Lt^CLIP+VF+S(θ)
            total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

            # Calculate gradients and update policy
            gradients = tape.gradient(total_loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

            return total_loss, policy_loss, value_loss


    def train(self, env, iterations=100, eval_interval=100, eval_episodes=100, csv_filename="training_results_farkel.csv"):
        """Main training loop following PPO algorithm with progress bars"""
        results_df = pd.DataFrame({
            'training_episode_index': pd.Series(dtype='int'),
            'mean_score': pd.Series(dtype='float'),
            'mean_time_per_episode': pd.Series(dtype='float'),
            'win_rate': pd.Series(dtype='float'),
            'mean_steps_per_episode': pd.Series(dtype='float'),
            'mean_time_per_step': pd.Series(dtype='float')
        })

        # Barre de progression principale pour les itérations
        with tqdm(total=iterations, desc="Training Progress") as pbar:
            for iteration in range(iterations):
                trajectories = []

                # Barre de progression pour la collecte des trajectoires
                for actor in tqdm(range(self.n_actors), desc=f"Collecting trajectories (Iteration {iteration + 1})",
                                  leave=False):
                    # Run policy πθ,a in environment for T timesteps
                    trajectory = self.run_actor(env)

                    # Compute advantage estimates Â1,...,ÂT
                    _, last_value = self.policy(tf.convert_to_tensor([trajectory['states'][-1]], dtype=tf.float32))
                    returns, advantages = self.compute_gae(
                        trajectory['rewards'],
                        trajectory['values'],
                        last_value.numpy()[0, 0],
                        trajectory['dones']
                    )
                    trajectory['returns'] = returns
                    trajectory['advantages'] = advantages
                    trajectories.append(trajectory)

                # Prepare data for optimization
                states = np.concatenate([t['states'] for t in trajectories])
                actions = np.concatenate([t['actions'] for t in trajectories])
                returns = np.concatenate([t['returns'] for t in trajectories])
                advantages = np.concatenate([t['advantages'] for t in trajectories])
                old_log_probs = np.concatenate([t['old_log_probs'] for t in trajectories])

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Barre de progression pour les epochs d'optimisation
                for epoch in tqdm(range(self.epochs), desc=f"Optimizing (Iteration {iteration + 1})", leave=False):
                    # Create minibatches of size M ≤ NT
                    indices = np.random.permutation(len(states))
                    for start_idx in range(0, len(states), self.batch_size):
                        batch_indices = indices[start_idx:start_idx + self.batch_size]

                        loss, policy_loss, value_loss = self.optimize_surrogate(
                            tf.convert_to_tensor(states[batch_indices], dtype=tf.float32),
                            tf.convert_to_tensor(actions[batch_indices], dtype=tf.int32),
                            tf.convert_to_tensor(old_log_probs[batch_indices], dtype=tf.float32),
                            tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32),
                            tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                        )

                # θold ← θ
                self.update_old_policy()

                if (iteration + 1) % eval_interval == 0:
                    eval_results = self.evaluate_policy(env, eval_episodes)

                    # Ajouter les résultats dans le DataFrame
                    results_df = pd.concat([
                        results_df,
                        pd.DataFrame([{
                            'training_episode_index': iteration + 1,
                            'mean_score': eval_results['mean_score'],
                            'mean_time_per_episode': eval_results['mean_time_per_episode'],
                            'win_rate': eval_results['win_rate'],
                            'mean_steps_per_episode': eval_results['mean_steps_per_episode'],
                            'mean_time_per_step': eval_results['mean_time_per_step']
                        }])
                    ], ignore_index=True)

                    # Sauvegarde dans un fichier CSV
                    results_df.to_csv(csv_filename, index=False)

                    print(f"\nÉvaluation : Épisode {iteration + 1}, Moyenne = {eval_results['mean_score']:.2f}, "
                          f"Longueur Moyenne = {eval_results['mean_steps_per_episode']:.2f}, Loss = {loss:.2f}, "
                          f"Policy Loss = {policy_loss:.2f}, "
                          f"Win Rate = {eval_results['win_rate']:.2f}")

                # Update progress bar
                pbar.update(1)

    def evaluate_policy(self, env, episodes=100):
        """Évalue la politique de l'agent et retourne uniquement les résultats nécessaires pour le CSV."""
        total_rewards = []
        total_lengths = []
        total_times = []
        total_steps = []
        win_count = 0

        for _ in range(episodes):
            start_time = time.time()

            env.reset()
            state = env.state_description()

            rewards = 0
            length = 0
            steps = 0
            done = False

            while not done:
                valid_actions = env.available_actions_ids()
                action, _ = self.select_action(state, valid_actions, test=True)

                prev_score = env.score()

                # Exécution de l'action
                env.step(action)
                reward = env.score() - prev_score
                next_state = env.state_description()
                done = env.is_game_over()

                rewards += reward
                state = next_state
                length += 1
                steps += 1

            # Temps pour cet épisode
            episode_time = time.time() - start_time
            total_rewards.append(rewards)
            total_lengths.append(length)
            total_times.append(episode_time)
            total_steps.append(steps)

            if rewards > 0:
                win_count += 1

        # Calcul des moyennes et des résultats spécifiques au CSV
        avg_reward = np.mean(total_rewards)
        avg_time_per_episode = np.mean(total_times)
        win_rate = win_count / episodes
        mean_steps_per_episode = np.mean(total_lengths)
        mean_time_per_step = avg_time_per_episode / mean_steps_per_episode if mean_steps_per_episode > 0 else 0

        # Retourne uniquement les données pertinentes pour le CSV
        return {
            'mean_score': avg_reward,
            'mean_time_per_episode': avg_time_per_episode,
            'win_rate': win_rate,
            'mean_steps_per_episode': mean_steps_per_episode,
            'mean_time_per_step': mean_time_per_step
        }



if __name__ == "__main__":
    env = FarkleDQNEnv(num_players=2, target_score=2000)
    #env = TicTacToe()
    agent = PPO_A2C_Style(
        state_dim=12,
        action_dim=128,
        n_actors=20,  # Plus d'acteurs parallèles
        timesteps=100,  # Plus de timesteps pour avoir des parties complètes
        epochs=10,
        batch_size=64,
        alpha=0.0001,
        gamma=0.99,
        clip_ratio=0.2,
        c1=0.5,
        c2=0.02
    )

    # Pour 500 itérations :
    # 20 acteurs × 100 timesteps = 2000 coups par itération
    # 2000 coups / 20 coups par partie ≈ 100 parties par itération
    # 100 parties × 500 itérations = 10000 parties

    agent.train(env, iterations=500)

    """ 
    16 × 256 = 4096
        expériences
        par
        itération
        5000
        itérations = ~20
        millions
        d
        'expériences totales
    """



    # Charger les données du CSV
    try:
        results = pd.read_csv("training_results_farkel.csv")
        print("Colonnes trouvées dans le fichier :", results.columns)
    except FileNotFoundError:
        print("Erreur : Le fichier 'training_results.csv' est introuvable.")
        exit(1)

    # Vérification des colonnes nécessaires
    required_columns = ['training_episode_index', 'mean_score', 'win_rate']
    missing_columns = [col for col in required_columns if col not in results.columns]

    if missing_columns:
        print(f"Colonnes manquantes dans le fichier : {missing_columns}")
        exit(1)

    # Tracer les métriques
    plt.figure(figsize=(10, 6))
    plt.plot(results['training_episode_index'], results['mean_score'], label='Score Moyen')
    plt.plot(results['training_episode_index'], results['win_rate'], label='Taux de Victoire')
    plt.xlabel('Épisodes')
    plt.ylabel('Valeurs')
    plt.title('Progression de la Formation')
    plt.legend()
    plt.show()
