import time
from statistics import mean
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from functions.outils import log_metrics_to_dataframe, plot_csv_data, save_files


class OneStepActorCritic:
    def __init__(self, state_dim, action_dim, alpha_theta=0.001, alpha_w=0.001, gamma=0.99, path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        self.path = path

        self.policy = self._build_policy()
        self.value = self._build_value()
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_theta, clipnorm=1.0)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_w, clipnorm=1.0)

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

    def _build_value(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer='glorot_normal',
                                  bias_initializer='zeros'),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer='glorot_normal',
                                  bias_initializer='zeros'),
            tf.keras.layers.Dense(1,
                                  kernel_initializer='glorot_normal',
                                  bias_initializer='zeros')
        ])

    def select_action(self, state_tensor, action_mask):
        action_mask = np.array(action_mask)
        q_s = self.policy(state_tensor[None])[0].numpy()
        masked_q_s = q_s * action_mask + np.finfo(float).eps * (1 - action_mask)
        return int(np.argmax(masked_q_s))

    def train_episode(self, env, episode_num):
        s = env.state_description()
        s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
        I = 1.0
        done = False
        total_reward = 0
        episode_steps = []

        while not done:
            action_mask = env.action_mask()

            action = self.select_action(s_tensor, action_mask)

            prev_score = env.score()
            env.step(action)
            next_state = env.state_description()
            reward = env.score() - prev_score
            done = env.is_game_over()

            next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                current_value = self.value(s_tensor[None])

                next_value = tf.zeros_like(current_value) if done else self.value(next_state_tensor[None])
                td_target = reward + self.gamma * next_value
                td_error = td_target - current_value
                value_loss = 0.5 * tf.reduce_mean(tf.square(td_error))

                # Policy network update with numerical stability
                action_probs = self.policy(s_tensor[None])
                action_mask_tensor = tf.convert_to_tensor(action_mask, dtype=tf.float32)
                masked_probs = action_probs * action_mask_tensor
                normalized_probs = masked_probs / (tf.reduce_sum(masked_probs, axis=1, keepdims=True) + 1e-8)

                action_one_hot = tf.one_hot(action, self.action_dim)
                log_prob = tf.math.log(tf.reduce_sum(normalized_probs * action_one_hot, axis=1) + 1e-8)
                policy_loss = -tf.reduce_mean(log_prob * tf.stop_gradient(td_error) * I)

            # Apply gradients with optimizers
            value_grads = tape.gradient(value_loss, self.value.trainable_variables)
            self.value_optimizer.apply_gradients(zip(value_grads, self.value.trainable_variables))

            policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

            del tape

            # Update state and accumulate rewards
            I *= self.gamma
            s_tensor = next_state_tensor
            total_reward += reward

            # Store metrics
            episode_steps.append({
                'value_loss': float(value_loss),
                'policy_loss': float(policy_loss),
                'td_error': float(td_error[0][0])
            })

        if not episode_steps:
            return total_reward, 0.0, 0.0

        return total_reward, np.mean([s['policy_loss'] for s in episode_steps]), np.mean(
            [s['value_loss'] for s in episode_steps])

    def train(self, env, episodes=100000, interval=1000):
        results_df = None

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            env.reset()
            total_reward, policy_loss, value_loss = self.train_episode(env, episode)

            if (episode + 1) % interval == 0 and episode > 0:
                results_df = log_metrics_to_dataframe(
                    function=play_with_actor_critic,
                    model=self.policy,
                    predict_func=None,
                    env=env,
                    episode_index=episode,
                    games=100,
                    dataframe=results_df
                )

                print(f"\n{'=' * 50}")
                print(f"Episode {episode + 1}")
                print(f"Policy Loss: {policy_loss:.6f}")
                print(f"Value Loss: {value_loss:.6f}")

        if self.path is not None:
            save_files(
                online_model=self.policy,
                algo_name="1_STEP_ACTOR_CRITIC",
                results_df=results_df,
                env=env,
                num_episodes=episodes,
                gamma=self.gamma,
                alpha=self.alpha_theta,
                optimizer=self.policy_optimizer,
                save_path=self.path
            )


        return results_df



def play_with_actor_critic(env, model, predict_func=None, episodes=100):
    """
    Fonction d'évaluation pour Actor-Critic suivant le même format que REINFORCE
    """
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
