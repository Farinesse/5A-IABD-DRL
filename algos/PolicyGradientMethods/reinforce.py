import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class REINFORCE:
    def __init__(self, state_dim, action_dim, alpha=0.0003, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.policy = self._build_policy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.reward_buffer = []

    def _build_policy(self):
        return keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_dim),
            keras.layers.Dense(256, activation='relu'),
            # keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_dim)
        ])

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)

        self.reward_buffer.extend(returns)
        if len(self.reward_buffer) > 1000:
            self.reward_buffer = self.reward_buffer[-1000:]

        # Normalisation
        if len(self.reward_buffer) > 0:
            returns = (returns - np.mean(self.reward_buffer)) / (np.std(self.reward_buffer) + 1e-8)
        return returns

    def train_episode(self, env):
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False

        # Liste pour stocker les time steps
        time_steps = []
        t = 0

        while not done:
            valid_actions = env.available_actions_ids()
            # Forward pass
            logits = self.policy(np.array(state).reshape(1, -1), training=False)
            logits = logits.numpy()[0]

            # Masquer les actions invalides
            mask = np.ones_like(logits) * float('-inf')
            mask[valid_actions] = 0
            masked_logits = logits + mask

            # Softmax avec température
            temperature = max(0.5, 1.0 - len(self.reward_buffer) / 5000)
            probs = tf.nn.softmax(masked_logits / temperature).numpy()

            # Sélection d'action
            if np.random.random() < 0.05:  # epsilon-greedy
                action = np.random.choice(valid_actions)
            else:
                action = np.random.choice(self.action_dim, p=probs)

            # Exécution de l'action
            env.step(action)
            reward = env.score() if env.is_game_over() else 0
            next_state = env.state_description()
            done = env.is_game_over()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            time_steps.append(t)  # Stockage du time step

            state = next_state
            t += 1  # Incrément du time step

        # Conversion en arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        time_steps = np.array(time_steps, dtype=np.float32)
        returns = self.compute_returns(rewards)

        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.policy(states, training=True)

            # Calcul des probabilités pour les actions choisies
            action_masks = tf.one_hot(actions, self.action_dim)
            selected_logits = tf.reduce_sum(logits * action_masks, axis=1)

            # Log probabilities
            log_probs = selected_logits - tf.reduce_logsumexp(logits, axis=1)

            # Application de gamma^t * G
            gamma_t = tf.pow(self.gamma, time_steps)  # Calcul de gamma^t
            time_discounted_returns = returns * gamma_t  # Application aux retours

            # Loss avec gamma^t
            basic_loss = -tf.reduce_mean(log_probs * time_discounted_returns)

            # Ajout de l'entropie
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            loss = basic_loss - 0.01 * entropy

        # Calcul et application des gradients
        grads = tape.gradient(loss, self.policy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        return sum(rewards), loss.numpy()

    def train(self, env, episodes=5000):
        history = []
        window_size = 100

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            total_reward, loss = self.train_episode(env)
            history.append(total_reward)

            if (episode + 1) % 100 == 0:
                recent_rewards = history[-window_size:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([r > 0 for r in recent_rewards])
                print(f"Episode {episode + 1}")
                print(f"Moyenne des récompenses: {avg_reward:.2f}")
                print(f"Taux de victoire: {win_rate:.2%}")
                print(f"Loss: {loss:.6f}\n")

        self.save_model('../../models/reinforce_model.h5')
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
            # Obtenir l'état actuel
            state = env.state_description()
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)

            # Obtenir les actions valides
            valid_actions = env.available_actions_ids()

            # Prédire les probabilités d'action
            logits = model(tf.expand_dims(state_tensor, 0), training=False)[0]

            # Masquer les actions invalides
            mask = np.ones_like(logits.numpy()) * float('-inf')
            mask[valid_actions] = 0
            masked_logits = logits + mask

            # Obtenir les probabilités
            probs = tf.nn.softmax(masked_logits).numpy()

            # Sélectionner l'action (en mode évaluation, on prend la meilleure action)
            if len(valid_actions) > 0:
                # Pendant le test, on prend l'action avec la plus haute probabilité
                action = valid_actions[np.argmax(probs[valid_actions])]
            else:
                print("Aucune action valide disponible!")
                break

            # Exécuter l'action
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            episode_reward += reward
            done = env.is_game_over()

            # Afficher l'état du jeu si demandé
            if display:
                print("\nÉtat actuel:")
                env.display()
                print(f"Action choisie: {action}")
                print(f"Récompense: {reward}")
                print(f"Score cumulé: {episode_reward}")
                print("Probabilités des actions:", probs[valid_actions])

        total_rewards += episode_reward
        print(f"\nÉpisode {episode + 1}/{episodes} terminé")
        print(f"Récompense totale de l'épisode: {episode_reward}")

    # Calculer et afficher le score moyen
    mean_score = total_rewards / episodes
    print(f"\nScore moyen sur {episodes} épisodes: {mean_score}")
    return mean_score


if __name__ == "__main__":
    from environment.tictactoe import TicTacToe
    from environment.FarkelEnv import FarkleDQNEnv

    # Configuration de TensorFlow pour moins de warnings
    tf.get_logger().setLevel('ERROR')

    env = TicTacToe()
    # env = FarkleDQNEnv()
    agent = REINFORCE(
        state_dim=27,
        action_dim=9,
        alpha=0.001,
        gamma=0.99
    )

    # history = agent.train(env, episodes=5000)
    model = keras.models.load_model('../../models/reinforce_model.h5')
    mean_score = play_with_reinforce(
        env=env,
        model=model,
        episodes=100,
        display=True
    )
