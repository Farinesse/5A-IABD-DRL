import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle
from functions.outils import save_files, play_with_reinforce_critic, log_metrics_to_dataframe


class REINFORCEWithCritic:
    def __init__(self, state_dim, action_dim, alpha_theta=0.0003, alpha_w=0.001, gamma=0.99, path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha_theta = alpha_theta  # Learning rate for policy
        self.alpha_w = alpha_w  # Learning rate for critic
        self.gamma = gamma
        self.path = path

        # Pour le suivi des performances
        self.reward_buffer = []

        # Initialize networks
        self.policy = self._build_policy()
        self.critic = self._build_critic()

        # Optimizers avec gradient clipping
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_theta, clipnorm=0.5)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_w, clipnorm=0.5)

    def _build_policy(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(self.action_dim)
        ])

    def _build_critic(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(1)
        ])

    def select_action(self, state_tensor, action_mask, valid_actions):
        """Sélection d'action avec masquage"""
        logits = self.policy(state_tensor[None])[0].numpy()

        mask = np.ones_like(logits) * float('-inf')
        mask[valid_actions] = 0

        probs = tf.nn.softmax(logits + mask).numpy()
        probs = probs / np.sum(probs)

        if not np.isclose(np.sum(probs), 1.0) or np.any(np.isnan(probs)):
            probs = np.zeros_like(logits)
            probs[valid_actions] = 1.0 / len(valid_actions)

        return int(np.random.choice(len(probs), p=probs))

    def compute_returns(self, rewards):
        """Calcul des retours avec normalisation"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns, dtype=np.float32)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        return returns

    def compute_shaped_reward(self, env, action, prev_score):
        """Calcul des récompenses avec shaping"""
        # Récompense de base
        reward = env.score() - prev_score

        if isinstance(env, TicTacToe):
            if reward == 0:  # Pas de victoire/défaite
                # Récompenses pour positions stratégiques
                strategic_positions = {4: 0.1}  # Centre
                corners = {0: 0.05, 2: 0.05, 6: 0.05, 8: 0.05}  # Coins

                if action in strategic_positions:
                    reward += strategic_positions[action]
                elif action in corners:
                    reward += corners[action]

        return reward

    def train_episode(self, env):
        states, actions, rewards = [], [], []
        state = env.state_description()
        done = False
        total_reward = 0
        timesteps = []
        t = 0

        while not done:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            action_mask = env.action_mask()
            valid_actions = env.available_actions_ids()

            # Sélectionner et exécuter l'action
            action = self.select_action(state_tensor, action_mask, valid_actions)
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score  # Utiliser uniquement la vraie récompense
            next_state = env.state_description()
            done = env.is_game_over()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            timesteps.append(t)

            state = next_state
            total_reward += reward
            t += 1

        # Convertir en tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        timesteps = tf.convert_to_tensor(timesteps, dtype=tf.float32)
        returns = self.compute_returns(rewards)

        # Update critic
        with tf.GradientTape() as tape:
            values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update policy
        with tf.GradientTape() as tape:
            # Calculer les avantages
            values = tf.squeeze(self.critic(states))
            advantages = returns - values

            # Pondération temporelle et clipping
            gamma_t = tf.pow(self.gamma, timesteps)
            # Dans train_episode
            advantages = tf.clip_by_value(advantages * gamma_t, -5.0, 5.0)  # -10,10 est trop large
            # Policy gradient avec baseline
            logits = self.policy(states)
            probabilities = tf.nn.softmax(logits)
            action_masks = tf.one_hot(actions, self.action_dim)

            # Log probs avec clipping numérique
            log_probs = tf.reduce_sum(
                tf.math.log(tf.clip_by_value(probabilities, 1e-10, 1.0)) * action_masks,
                axis=1
            )

            policy_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))

        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        self.reward_buffer.append(total_reward)
        return total_reward, float(policy_loss), float(critic_loss)

    def train(self, env, episodes=10000, interval=1000):
        results_df = None

        for episode in tqdm(range(episodes), desc="Training"):
            env.reset()
            total_reward, policy_loss, critic_loss = self.train_episode(env)

            if (episode + 1) % interval == 0:


                results_df = log_metrics_to_dataframe(
                    function=play_with_reinforce_critic,
                    model=self.policy,
                    predict_func=None,
                    env=env,
                    episode_index=episode,
                    games=1000,
                    dataframe=results_df
                )

        # Utiliser save_files à la fin de l'entraînement
        if self.path is not None:
            save_files(
                online_model=self.policy,
                algo_name="REINFORCE_CRITIC",
                results_df=results_df,
                env=env,
                num_episodes=episodes,
                gamma=self.gamma,
                alpha=self.alpha_theta,  # Learning rate de la politique
                optimizer=self.policy_optimizer,
                save_path=self.path,
                custom_metrics={
                    'critic_learning_rate': self.alpha_w,
                    'critic_loss': critic_loss
                }
            )

        return results_df



    def get_weights(self):
        """Récupère les poids pour la sauvegarde"""
        return {
            'policy_weights': self.policy.get_weights(),
            'critic_weights': self.critic.get_weights(),
            'policy_optimizer_config': self.policy_optimizer.get_config(),
            'critic_optimizer_config': self.critic_optimizer.get_config()
        }

    def set_weights(self, weights):
        """Charge les poids depuis une sauvegarde"""
        self.policy.set_weights(weights['policy_weights'])
        self.critic.set_weights(weights['critic_weights'])
        self.policy_optimizer = tf.keras.optimizers.Adam.from_config(weights['policy_optimizer_config'])
        self.critic_optimizer = tf.keras.optimizers.Adam.from_config(weights['critic_optimizer_config'])




    def save_model(self, file_path):
        """Sauvegarde le modèle avec Pickle (.pkl)"""
        try:
            model_state = {
                'weights': self.get_weights(),
                'hyperparameters': {
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'alpha_theta': self.alpha_theta,
                    'alpha_w': self.alpha_w,
                    'gamma': self.gamma
                }
            }
            with open(file_path, 'wb') as f:
                pickle.dump(model_state, f)
            print(f"Modèle sauvegardé avec succès dans {file_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle : {e}")


    def load_model(file_path):
        """Charge un modèle sauvegardé (.pkl)"""
        try:
            with open(file_path, 'rb') as f:
                model_state = pickle.load(f)

            model = REINFORCEWithCritic(
                state_dim=model_state['hyperparameters']['state_dim'],
                action_dim=model_state['hyperparameters']['action_dim'],
                alpha_theta=model_state['hyperparameters']['alpha_theta'],
                alpha_w=model_state['hyperparameters']['alpha_w'],
                gamma=model_state['hyperparameters']['gamma']
            )

            model.set_weights(model_state['weights'])
            print(f"Modèle chargé avec succès depuis {file_path}")
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            return None
