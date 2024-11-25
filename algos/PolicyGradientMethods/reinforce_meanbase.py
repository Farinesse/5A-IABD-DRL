import os
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, alpha_theta=0.001, alpha_w=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma

        # Create checkpoint directories
        self.checkpoint_dir = 'checkpoints_actor'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Policy network
        self.policy = self._build_policy()
        # Value network
        self.baseline = self._build_baseline()

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_theta)
        self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_w)

    def _build_policy(self):
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.action_dim)
        ])

    def _build_baseline(self):
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),

            keras.layers.Dense(1)
        ])

    def save_checkpoint(self, episode, metrics):
        """
        Save metrics and models with error handling and backup mechanism
        """
        try:
            # Save metrics first
            df = pd.DataFrame([{
                'episode': episode,
                **metrics
            }])

            metrics_file = 'evaluation_metrics.csv'
            df.to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file), index=False)

            # Save models with keras format instead of HDF5
            policy_path = os.path.join(self.checkpoint_dir, f'policy_{episode}.keras')
            baseline_path = os.path.join(self.checkpoint_dir, f'baseline_{episode}.keras')

            # Save temporary files first
            temp_policy_path = policy_path + '.tmp'
            temp_baseline_path = baseline_path + '.tmp'

            self.policy.save(temp_policy_path, save_format='keras')
            self.baseline.save(temp_baseline_path, save_format='keras')

            # If saves were successful, rename to final files
            if os.path.exists(temp_policy_path):
                if os.path.exists(policy_path):
                    os.remove(policy_path)
                os.rename(temp_policy_path, policy_path)

            if os.path.exists(temp_baseline_path):
                if os.path.exists(baseline_path):
                    os.remove(baseline_path)
                os.rename(temp_baseline_path, baseline_path)

        except Exception as e:
            print(f"Warning: Failed to save checkpoint at episode {episode}. Error: {str(e)}")
            # Clean up any temporary files
            for temp_file in [temp_policy_path, temp_baseline_path]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

    def evaluate(self, env, n_episodes=1000):
        """
        Évalue le modèle sur n_episodes sans exploration
        """
        total_rewards = []
        total_steps = []

        for _ in range(n_episodes):
            env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                state = env.state_description()
                valid_actions = env.available_actions_ids()

                # Forward pass pour obtenir les logits
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                logits = self.policy(state_tensor, training=False)[0].numpy()

                # Masquer les actions invalides
                mask = np.ones_like(logits) * float('-inf')
                mask[valid_actions] = 0
                masked_logits = logits + mask

                # Prendre l'action la plus probable (pas d'exploration)
                probs = tf.nn.softmax(masked_logits).numpy()
                action = valid_actions[np.argmax(probs[valid_actions])]


                env.step(action)
                reward = env.score() - prev_score
                if env.is_game_over():
                    episode_reward += env.score()

                steps += 1

            total_rewards.append(episode_reward)
            total_steps.append(steps)

        metrics = {
            'avg_reward': np.mean(total_rewards),
            'win_rate': np.mean([r > 0 for r in total_rewards]),
            'avg_steps': np.mean(total_steps)
        }

        return metrics

    def plot_metrics(self, metrics_file='evaluation_metrics.csv'):
        """
        Trace les courbes d'évolution des métriques
        """
        try:
            df = pd.read_csv(metrics_file)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

            # Plot average reward
            ax1.plot(df['episode'], df['avg_reward'])
            ax1.set_title('Récompense moyenne')
            ax1.set_xlabel('Épisodes')
            ax1.set_ylabel('Récompense')
            ax1.grid(True)

            # Plot win rate
            ax2.plot(df['episode'], df['win_rate'])
            ax2.set_title('Taux de victoire')
            ax2.set_xlabel('Épisodes')
            ax2.set_ylabel('Taux de victoire')
            ax2.grid(True)

            # Plot average steps
            ax3.plot(df['episode'], df['avg_steps'])
            ax3.set_title('Nombre moyen de coups')
            ax3.set_xlabel('Épisodes')
            ax3.set_ylabel('Coups')
            ax3.grid(True)

            plt.tight_layout()
            plt.savefig('learning_curves.png')
            plt.close()

        except Exception as e:
            print(f"Erreur lors du tracé des métriques : {str(e)}")
    def train(self, env, episodes=1000000, eval_frequency=1000, eval_episodes=1000):
        """
        Train the model with periodic evaluation and robust checkpoint saving
        """
        history = defaultdict(list)

        # Keep track of best performance for save on improvement
        best_win_rate = -float('inf')

        try:
            for episode in tqdm(range(episodes), desc="Training"):
                env.reset()
                episode_reward, loss = self.train_episode(env)
                history['rewards'].append(episode_reward)

                # Periodic evaluation
                if (episode + 1) % eval_frequency == 0:
                    print(f"\nÉvaluation à l'épisode {episode + 1}...")
                    metrics = self.evaluate(env, n_episodes=eval_episodes)

                    print(f"Moyenne des récompenses: {metrics['avg_reward']:.2f}")
                    print(f"Taux de victoire: {metrics['win_rate']:.2%}")
                    print(f"Nombre moyen de coups: {metrics['avg_steps']:.1f}")

                    # Save checkpoint if performance improved
                    if metrics['win_rate'] > best_win_rate:
                        best_win_rate = metrics['win_rate']
                        self.save_checkpoint(episode + 1, metrics)

                    # Plot learning curves
                    try:
                        self.plot_metrics()
                    except Exception as e:
                        print(f"Warning: Failed to plot metrics. Error: {str(e)}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving final checkpoint...")
            metrics = self.evaluate(env, n_episodes=eval_episodes)
            self.save_checkpoint(episode + 1, metrics)

        return history
    def compute_G(self, rewards, t):
        """Compute Gt = Σᵏ₌ₜ₊₁ᵀ γᵏ⁻ᵗ⁻¹Rₖ"""
        G = 0
        for k in range(t + 1, len(rewards)):
            G += (self.gamma ** (k - t - 1)) * rewards[k]
        return G
    def train_episode(self, env):
        # Generate an episode S₀,A₀,R₁,...,Sₜ₋₁,Aₜ₋₁,Rₜ, following π(·|·,θ)
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False

        while not done:
            valid_actions = env.available_actions_ids()

            # Get action probabilities from policy π(·|s,θ)
            state_tensor = tf.convert_to_tensor(np.array(state).reshape(1, -1), dtype=tf.float32)
            probs = self.policy(state_tensor, training=False)[0].numpy()

            # Mask invalid actions
            mask = np.ones_like(probs) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(probs + mask).numpy()

            # Sample action from policy
            action = np.random.choice(self.action_dim, p=masked_probs)

            # Execute action
            env.step(action)
            reward = env.score() if env.is_game_over() else 0
            next_state = env.state_description()
            done = env.is_game_over()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Loop for each step of the episode t = 0,1,...,T-1:
        total_loss = 0
        for t in range(len(states)):
            state_t = np.array(states[t], dtype=np.float32).reshape(1, -1)
            action_t = actions[t]

            # G ← Σᵏ₌ₜ₊₁ᵀ γᵏ⁻ᵗ⁻¹Rₖ
            G = self.compute_G(rewards, t)

            # δ ← G - v̂(Sₜ,w)
            with tf.GradientTape() as tape:
                baseline_value = self.baseline(state_t)
                delta = G - tf.squeeze(baseline_value)
                # w ← w + αᵂδ∇v̂(Sₜ,w)
                baseline_loss = tf.square(delta)

            baseline_grads = tape.gradient(baseline_loss, self.baseline.trainable_variables)
            self.baseline_optimizer.apply_gradients(zip(baseline_grads, self.baseline.trainable_variables))

            # θ ← θ + αθγᵗδ∇ln π(Aₜ|Sₜ,θ)
            with tf.GradientTape() as tape:
                logits = self.policy(state_t)
                action_mask = tf.one_hot([action_t], self.action_dim)
                log_prob = tf.reduce_sum(tf.math.log(logits + 1e-10) * action_mask)
                policy_loss = -(self.gamma ** t) * delta * log_prob

            policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

            total_loss += policy_loss.numpy()

        return sum(rewards), total_loss

    def save_models(self, policy_path, baseline_path):
        self.policy.save(policy_path)
        self.baseline.save(baseline_path)


# Configuration optimisée pour Farkel
if __name__ == "__main__":
    from environment.FarkelEnv import FarkleDQNEnv

    # Configuration de l'environnement TensorFlow
    tf.get_logger().setLevel('ERROR')

    # Création de l'environnement et de l'agent
    env = FarkleDQNEnv(target_score=5000)
    agent = REINFORCEWithBaseline(
        state_dim=12,
        action_dim=128,
        alpha_theta=0.00005,  # Learning rate plus petit pour plus de stabilité
        alpha_w=0.0005,  # Learning rate du baseline aussi réduit
        gamma=0.995  # Gamma plus élevé pour mieux considérer les récompenses futures
    )

    # Paramètres d'entraînement ajustés
    history = agent.train(
        env,
        episodes=20000,  # 500K épisodes devraient suffire
        eval_frequency=1000,  # Évaluation moins fréquente
        eval_episodes=100  # Plus d'épisodes d'évaluation pour des métriques plus stables
    )

    agent.plot_metrics(metrics_file="evaluation_metrics_reinforce.csv")

