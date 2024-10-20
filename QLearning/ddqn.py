import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm

@tf.function(reduce_retracing=True)
def gradient_step(model, s, a, target, optimizer):
    with tf.GradientTape() as tape:
        s = tf.ensure_shape(s, [27])  # Dynamically use input_dim
        a = tf.cast(a, dtype=tf.int32)
        q_s_a = model(tf.expand_dims(s, 0))[0][a]
        loss = tf.square(q_s_a - target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function(reduce_retracing=True)
def model_predict(model, s):
    s = tf.ensure_shape(s, [None])  # Ensure constant shape
    return model(tf.expand_dims(s, 0))[0]

def epsilon_greedy_action(
    q_s: tf.Tensor,
    mask: tf.Tensor,
    available_actions: np.ndarray,
    epsilon: float
) -> int:
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        inverted_mask = tf.constant(1.0) - mask
        masked_q_s = q_s * mask + tf.float32.min * inverted_mask
        return int(tf.argmax(masked_q_s, axis=0))

def save_model(model, file_path):
    try:
        model.save(file_path)
        print(f"Model successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")

def double_dqn_no_replay(online_model, target_model, env, num_episodes, gamma, alpha, start_epsilon, end_epsilon,
                         update_target_steps=10000, save_path='double_dqn_model_tictactoe.h5'):
    optimizer = keras.optimizers.SGD(learning_rate=alpha, momentum=0.9, nesterov=True)

    epsilon = start_epsilon
    total_score = 0.0
    total_loss = 0.0

    for ep_id in tqdm(range(num_episodes)):
        if ep_id % 1000 == 0 and ep_id > 0:
            print(f"Mean Score: {total_score / 1000}, Mean Loss: {total_loss / 1000}, Epsilon: {epsilon}")
            total_score = 0.0
            total_loss = 0.0

        env.reset()
        if env.is_game_over():
            continue

        while not env.is_game_over():
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_s = model_predict(online_model, s_tensor)
            a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), epsilon)

            if a not in env.available_actions_ids():
                print(f"Invalid action {a}, taking random action instead.")
                a = np.random.choice(env.available_actions_ids())

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            if env.is_game_over():
                target = r
            else:
                s_prime = env.state_description()
                s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)
                next_mask = env.action_mask()
                next_mask_tensor = tf.convert_to_tensor(next_mask, dtype=tf.float32)

                # Double Q-learning update
                q_next_online = model_predict(online_model, s_prime_tensor)
                best_action = tf.argmax(q_next_online * next_mask_tensor)
                q_next_target = model_predict(target_model, s_prime_tensor)
                target = r + gamma * q_next_target[best_action]

            loss = gradient_step(online_model, s_tensor, a, target, optimizer)
            total_loss += loss.numpy()

        total_score += env.score()
        progress = ep_id / num_episodes
        epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        if ep_id % update_target_steps == 0:
            target_model.set_weights(online_model.get_weights())

    # Save the online model at the end of training
    save_model(online_model, save_path)

    return online_model, target_model