import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense
from collections import deque
import random
from tqdm import tqdm
from environment.farkel_env import FarkleDQNEnv


@tf.function(reduce_retracing=True)
def gradient_step(model, s, a, target, optimizer):
    with tf.GradientTape() as tape:
        s = tf.ensure_shape(s, [10])  # Dynamically use input_dim
        a = tf.cast(a, dtype=tf.int32)
        q_s_a = model(tf.expand_dims(s, 0))[0][a]
        loss = tf.square(q_s_a - target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function(reduce_retracing=True)
def model_predict(model, s):
    s = tf.ensure_shape(s, [None])  # Assurer une forme constante
    return model(tf.expand_dims(s, 0))[0]


def epsilon_greedy_action(q_s: tf.Tensor, mask: tf.Tensor, available_actions: np.ndarray, epsilon: float) -> int:
    """
    Choisit une action avec une politique epsilon-greedy en s'assurant de ne pas choisir d'actions invalides.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        masked_q_s = q_s * mask + (1.0 - mask) * tf.float32.min
        action = int(tf.argmax(masked_q_s, axis=0))
        if action not in available_actions:
            action = np.random.choice(available_actions)
        return action


def create_dqn_model(input_dim=10, output_dim=128):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def deep_q_learning(model, target_model, env, num_episodes, gamma, alpha, start_epsilon, end_epsilon,
                    memory_size=5000, batch_size=32, update_target_steps=500, epsilon_decay=0.999):
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    memory = deque(maxlen=memory_size)
    epsilon = start_epsilon
    total_score = 0.0
    total_loss = 0.0

    for ep_id in tqdm(range(num_episodes)):
        if ep_id % 100 == 0 and ep_id > 0:
            print(f"Mean Score: {total_score / 100}, Mean Loss: {total_loss / 100}, Epsilon: {epsilon}")
            total_score = 0.0
            total_loss = 0.0

        state = env.reset()
        if env.is_game_over():
            continue

        while not env.is_game_over():
            env.roll_dice(env.remaining_dice)
            state = env.get_observation()
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_s = model_predict(model, state_tensor)
            action = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), epsilon)

            prev_score = env.scores[env.current_player]
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)

            memory.append((state_tensor, action, reward, next_state_tensor, done))

            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)

                for state_mb, action_mb, reward_mb, next_state_mb, done_mb in minibatch:
                    if done_mb:
                        target = reward_mb
                    else:
                        next_mask = env.action_mask()
                        next_mask_tensor = tf.convert_to_tensor(next_mask, dtype=tf.float32)
                        q_next = model_predict(target_model, next_state_mb)
                        target = reward_mb + gamma * tf.reduce_max(q_next * next_mask_tensor)

                    loss = gradient_step(model, state_mb, action_mb, target, optimizer)
                    total_loss += loss.numpy()

        total_score += env.scores[env.current_player]
        epsilon = max(end_epsilon, epsilon * epsilon_decay)

        if ep_id % update_target_steps == 0:
            target_model.set_weights(model.get_weights())

    return model


def train_farkle_dqn(num_episodes=10000):
    env = FarkleDQNEnv()

    input_dim = len(env.get_observation())
    model = create_dqn_model(input_dim=input_dim)
    target_model = create_dqn_model(input_dim=input_dim)
    target_model.set_weights(model.get_weights())

    return deep_q_learning(
        model=model,
        target_model=target_model,
        env=env,
        num_episodes=num_episodes,
        gamma=0.99,
        alpha=0.001,
        start_epsilon=1.0,
        end_epsilon=0.01,
        memory_size=10000,
        batch_size=32,
        update_target_steps=500,
        epsilon_decay=0.999
    )


def evaluate_model(model, num_games=100):
    env = FarkleDQNEnv()
    scores = []

    for _ in range(num_games):
        state = env.reset()
        done = False

        while not done:
            env.roll_dice(env.remaining_dice)
            state = env.get_observation()
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_values = model_predict(model, state_tensor)
            action = epsilon_greedy_action(q_values, mask_tensor, env.available_actions_ids(), epsilon=0)

            _, _, done, _ = env.step(action)

        scores.append(env.scores[0])

    return np.mean(scores), np.std(scores)


if __name__ == "__main__":
    trained_model = train_farkle_dqn(num_episodes=100)

    mean_score, std_score = evaluate_model(trained_model, num_games=100)
    print(f"Évaluation du modèle sur 100 parties:")
    print(f"Score moyen: {mean_score:.2f} ± {std_score:.2f}")

    # Démonstration d'une partie
    env = FarkleDQNEnv()
    state = env.reset()
    done = False

    while not done:
        env.roll_dice(env.remaining_dice)
        state = env.get_observation()
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        mask = env.action_mask()
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

        q_values = model_predict(trained_model, state_tensor)
        action = epsilon_greedy_action(q_values, mask_tensor, env.available_actions_ids(), epsilon=0)

        _, reward, done, _ = env.step(action)
        env.render()

    print(f"Partie terminée! Score final: {env.scores[0]}")