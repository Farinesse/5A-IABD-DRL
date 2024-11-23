import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import pandas as pd
from collections import defaultdict


class REINFORCEWithCritic:
    def __init__(self, state_dim, action_dim, alpha_policy=0.001, alpha_critic=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Construction des réseaux
        self.policy = self._build_policy()
        self.critic = self._build_critic()

        # Optimiseurs séparés pour la politique et le critique
        initial_learning_rate = alpha_policy
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_critic)

    def _build_policy(self):
        return keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(512, activation='relu'),  # Couche supplémentaire
            keras.layers.Dropout(0.1),
            keras.layers.Dense(self.action_dim)  # 128 actions possibles
        ])

    def _build_critic(self):
        return keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1)
        ])

    def select_action(self, state, valid_actions):
        # Epsilon-greedy pour l'exploration
        if np.random.random() < 0.05:
            return np.random.choice(valid_actions)

        # Forward pass pour obtenir les logits
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.policy(state_tensor, training=False)[0].numpy()

        # Masquer les actions invalides
        mask = np.ones_like(logits) * float('-inf')
        mask[valid_actions] = 0
        masked_logits = logits + mask

        # Softmax avec température adaptative
        temperature = max(0.5, 1.0)  # Température fixe ou adaptative
        probs = tf.nn.softmax(masked_logits / temperature).numpy()

        return np.random.choice(self.action_dim, p=probs)

    def compute_returns(self, rewards):
        # Calcul des retours Monte Carlo
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return np.array(returns, dtype=np.float32)

    def train_episode(self, env):
        # Collecte de trajectoire
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False

        while not done:
            # Sélection et exécution de l'action
            valid_actions = env.available_actions_ids()
            action = self.select_action(state, valid_actions)

            # Exécution de l'action
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            next_state = env.state_description()
            done = env.is_game_over()

            # Stockage des transitions
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Conversion en tensors
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        state_tensors = tf.convert_to_tensor(states)

        # Calcul des retours Monte Carlo
        returns = self.compute_returns(rewards)

        # Mise à jour du critique (baseline)
        with tf.GradientTape() as critic_tape:
            # Prédiction des valeurs d'état
            values = self.critic(state_tensors)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))

        # Application des gradients du critique
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Calcul des avantages (retours - baseline)
        advantages = returns - tf.squeeze(values)

        # Mise à jour de la politique
        with tf.GradientTape() as policy_tape:
            # Forward pass de la politique
            logits = self.policy(state_tensors)

            # Calcul des probabilités des actions prises
            action_masks = tf.one_hot(actions, self.action_dim)
            selected_logits = tf.reduce_sum(logits * action_masks, axis=1)

            # Log probabilities
            log_probs = selected_logits - tf.reduce_logsumexp(logits, axis=1)

            # Loss de la politique avec avantages
            policy_loss = -tf.reduce_mean(log_probs * advantages)

            # Ajout d'un terme d'entropie pour encourager l'exploration
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            policy_loss = policy_loss - 0.001 * entropy

        # Application des gradients de la politique
        policy_grads = policy_tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        return sum(rewards), policy_loss.numpy(), critic_loss.numpy()

    def train(self, env, episodes=10000):
        history = {
            'rewards': [],
            'policy_losses': [],
            'critic_losses': [],
            'steps_per_episode': [],
            'time_per_step': [],
            'win_rates': []
        }
        window_size = 100
        metrics_checkpoints = [1000, 10000, 100000, 1000000]

        # Pour le calcul du temps moyen par coup
        total_steps = 0
        total_time = 0

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            start_time = time.time()

            env.reset()
            total_reward, policy_loss, critic_loss = self.train_episode(env)

            episode_time = time.time() - start_time
            steps_this_episode = len(history['steps_per_episode']) + 1
            total_steps += steps_this_episode
            total_time += episode_time

            # Stockage des métriques
            history['rewards'].append(total_reward)
            history['policy_losses'].append(policy_loss)
            history['critic_losses'].append(critic_loss)
            history['steps_per_episode'].append(steps_this_episode)
            history['time_per_step'].append(episode_time / steps_this_episode)

            if (episode + 1) % 100 == 0:
                recent_rewards = history['rewards'][-window_size:]
                recent_steps = history['steps_per_episode'][-window_size:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([r > 0 for r in recent_rewards])
                avg_steps = np.mean(recent_steps)
                avg_time_per_step = np.mean(history['time_per_step'][-window_size:])

                print(f"\n{'=' * 50}")
                print(f"Episode {episode + 1}")
                print(f"{'=' * 50}")
                print(f"Métriques de performance:")
                print(f"  Moyenne des récompenses: {avg_reward:.2f}")
                print(f"  Taux de victoire: {win_rate:.2%}")
                print(f"  Nombre moyen de coups: {avg_steps:.1f}")
                print(f"  Temps moyen par coup: {avg_time_per_step * 1000:.2f}ms")
                print(f"\nMétriques d'apprentissage:")
                print(f"  Policy Loss: {policy_loss:.6f}")
                print(f"  Critic Loss: {critic_loss:.6f}")

            # Sauvegarde aux checkpoints_actor spécifiques
            if (episode + 1) in metrics_checkpoints:
                self.save_checkpoint_metrics(history, episode + 1)
                self.save_models(f'reinforce_policy_{episode + 1}.h5', f'reinforce_critic_{episode + 1}.h5')

        # Sauvegarde finale
        self.save_checkpoint_metrics(history, episodes)
        self.save_models('reinforce_policy_final.h5', 'reinforce_critic_final.h5')

        return history

    def save_models(self, policy_path, critic_path):
        """
        Sauvegarde les modèles policy et critic dans le dossier checkpoints_actor
        """
        try:
            # Chemins complets dans le dossier checkpoints_actor
            policy_full_path = os.path.join('checkpoints_actor', policy_path)
            critic_full_path = os.path.join('checkpoints_actor', critic_path)

            # Sauvegarde des modèles
            self.policy.save(policy_full_path)
            self.critic.save(critic_full_path)
            print(f"Modèles sauvegardés dans :")
            print(f"  - Policy: {policy_full_path}")
            print(f"  - Critic: {critic_full_path}")

        except Exception as e:
            print(f"Erreur lors de la sauvegarde des modèles: {str(e)}")
    def save_checkpoint_metrics(self, history, episode):
        metrics = {
            'episode': episode,
            'avg_reward': np.mean(history['rewards'][-1000:]),
            'win_rate': np.mean([r > 0 for r in history['rewards'][-1000:]]),
            'avg_steps': np.mean(history['steps_per_episode'][-1000:]),
            'avg_time_per_step': np.mean(history['time_per_step'][-1000:]) * 1000,  # en ms
            'policy_loss': np.mean(history['policy_losses'][-1000:]),
            'critic_loss': np.mean(history['critic_losses'][-1000:])
        }

        # Sauvegarde dans un fichier CSV
        df = pd.DataFrame([metrics])
        filename = 'metrics.csv'
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

        print(f"\nMétriques au checkpoint {episode}:")
        print(f"{'=' * 50}")
        print(f"Score moyen: {metrics['avg_reward']:.2f}")
        print(f"Taux de victoire: {metrics['win_rate']:.2%}")
        print(f"Longueur moyenne partie: {metrics['avg_steps']:.1f}")
        print(f"Temps moyen par coup: {metrics['avg_time_per_step']:.2f}ms")


if __name__ == "__main__":
    from environment.FarkelEnv import FarkleDQNEnv
    import os

    # Création du dossier pour les sauvegardes
    os.makedirs('checkpoints_actor', exist_ok=True)

    tf.get_logger().setLevel('ERROR')

    env = FarkleDQNEnv(target_score=2000)
    agent = REINFORCEWithCritic(
        state_dim=12,  # Taille de votre state_description
        action_dim=128,  # Nombre d'actions possibles dans Farkel
        alpha_policy=0.0001,  # Learning rate plus petit pour stabilité
        alpha_critic=0.001,  # Learning rate plus grand pour le critic
        gamma=0.99  # Discount factor
    )

    print("Début de l'entraînement sur 10000 épisodes...")
    history = agent.train(env, episodes=100000)
    print("\nEntraînement terminé!")

    # Affichage des métriques finales
    print("\nMétriques finales:")
    print("=" * 50)
    print(f"Score moyen final: {np.mean(history['rewards'][-1000:]):.2f}")
    print(f"Taux de victoire final: {np.mean([r > 0 for r in history['rewards'][-1000:]]):.2%}")
    print(f"Longueur moyenne partie: {np.mean(history['steps_per_episode'][-1000:]):.1f}")
    print(f"Temps moyen par coup: {np.mean(history['time_per_step'][-1000:]) * 1000:.2f}ms")

