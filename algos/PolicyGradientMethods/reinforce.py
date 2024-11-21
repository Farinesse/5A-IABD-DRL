import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class REINFORCE:
    def __init__(self, state_dim, action_dim, alpha=0.0001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha  # step size α
        self.gamma = gamma  # discount factor
        self.policy = self._build_policy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.reward_buffer = []

    def _build_policy(self):
        # π(a|s,θ) - policy parameterization
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.action_dim, activation='softmax')  # Sortie en distribution de probabilités
        ])

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

            # π(a|s,θ) - Calcul des probabilités d'action
            probs = self.policy(state_tensor, training=False)[0].numpy()

            # Masquer les actions invalides
            mask = np.ones_like(probs) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(probs + mask).numpy()

            # Sélection d'action selon π(a|s,θ)
            epsilon = max(0.01, 0.1 * (1 - len(self.reward_buffer) / 8000))  # Décroissance d'epsilon
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
        history = []
        window_size = 500

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            total_reward, loss = self.train_episode(env)
            history.append(total_reward)

            if (episode + 1) % 100 == 0:
                recent_rewards = history[-window_size:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([r > 0 for r in recent_rewards])
                print(f"\nEpisode {episode + 1}")
                print(f"Moyenne des récompenses: {avg_reward:.2f}")
                print(f"Taux de victoire: {win_rate:.2%}")
                print(f"Loss: {loss:.6f}")

        self.save_model('tiktactoe_reinforce_model.h5')
        return history

    def save_model(self, filepath):
        self.policy.save(filepath)


def play_with_reinforce(env, model, episodes=1, display=True):
    total_rewards = 0

    for episode in range(episodes):
        env.reset()
        done = False
        episode_reward = 0

        while not done:
            state = env.state_description()
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            valid_actions = env.available_actions_ids()

            probs = model(tf.expand_dims(state_tensor, 0), training=False)[0]
            mask = np.ones_like(probs.numpy()) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(probs + mask).numpy()

            # En évaluation, on prend l'action la plus probable
            if len(valid_actions) > 0:
                action = valid_actions[np.argmax(masked_probs[valid_actions])]
            else:
                print("Aucune action valide disponible!")
                break

            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            episode_reward += reward
            done = env.is_game_over()

            if display:
                print("\nÉtat actuel:")
                env.display()
                print(f"Action choisie: {action}")
                print(f"Récompense: {reward}")
                print(f"Score cumulé: {episode_reward}")
                print("Probabilités des actions:", masked_probs[valid_actions])

        total_rewards += episode_reward
        print(f"\nÉpisode {episode + 1}/{episodes} terminé")
        print(f"Récompense totale de l'épisode: {episode_reward}")

    mean_score = total_rewards / episodes
    print(f"\nScore moyen sur {episodes} épisodes: {mean_score}")
    return mean_score


if __name__ == "__main__":
    from environment.tictactoe import TicTacToe

    tf.get_logger().setLevel('ERROR')

    env = TicTacToe()
    agent = REINFORCE(
        state_dim=27,
        action_dim=9,
        alpha=0.001,
        gamma=0.99
    )

    history = agent.train(env, episodes=10000)
    env.reset()
    model = keras.models.load_model('reinforce_model.h5')
    mean_score = play_with_reinforce(env=env, model=model, episodes=100, display=True)