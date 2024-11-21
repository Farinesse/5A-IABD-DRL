import random
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import deque
from functions.outils import custom_two_phase_decay


@tf.function(reduce_retracing=True)
def gradient_step(model, s, a, target, optimizer):
    with tf.GradientTape() as tape:
        # Assurez-vous que s est traité comme un batch
        a = tf.cast(a, dtype=tf.int32)
        q_values = model(s)
        # Utiliser les actions pour récupérer les Q-values correspondantes
        q_s_a = tf.gather_nd(q_values, tf.expand_dims(a, axis=1), batch_dims=1)
        loss = tf.reduce_mean(tf.square(q_s_a - target))  # Moyenne de la perte sur le batch
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss



@tf.function(reduce_retracing=True)
def model_predict(
        model,
        s
):
    s = tf.ensure_shape(s, [None, None])  # Ensure constant shape
    return model(s)


def epsilon_greedy_action(
        q_s,
        mask,
        available_actions,
        epsilon
):
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        inverted_mask = tf.constant(1.0) - mask
        masked_q_s = q_s * mask + tf.float32.min * inverted_mask
        return int(tf.argmax(masked_q_s, axis=-1))


def save_model(
        model,
        file_path
):
    try:
        model.save(file_path)
        print(f"Model successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")


def double_dqn_with_replay(
        online_model,
        target_model,
        env,
        num_episodes,
        gamma,
        alpha,
        start_epsilon,
        end_epsilon,
        update_target_steps=10000,
        batch_size=32,
        memory_size=10000,
        save_path='double_dqn_with_exp_rep_model_tictactoe.h5'
):
    optimizer = keras.optimizers.SGD(learning_rate=alpha, momentum=0.9, nesterov=True)

    epsilon = start_epsilon
    total_score = 0.0
    total_loss = 0.0
    memory = deque(maxlen=memory_size)

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

            q_s = model_predict(online_model, tf.expand_dims(s_tensor, 0))[0]
            a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), epsilon)

            if a not in env.available_actions_ids():
                print(f"Invalid action {a}, taking random action instead.")
                a = np.random.choice(env.available_actions_ids())

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)

            # Store experience in memory
            memory.append((s_tensor, a, r, s_prime_tensor, env.is_game_over()))

            # Experience Replay
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = tf.stack(states)
                next_states = tf.stack(next_states)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                dones = tf.convert_to_tensor(dones, dtype=tf.float32)

                # Double Q-learning update
                q_next_online = model_predict(online_model, next_states)
                q_next_target = model_predict(target_model, next_states)

                best_actions = tf.argmax(q_next_online, axis=1)
                best_actions_one_hot = tf.one_hot(best_actions, depth=q_next_target.shape[-1])
                next_q_values = tf.reduce_sum(q_next_target * best_actions_one_hot, axis=1)

                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = gradient_step(online_model, states, actions, targets, optimizer)
                total_loss += loss.numpy()

        total_score += env.score()
        progress = ep_id / num_episodes
        epsilon = max(end_epsilon, start_epsilon - (start_epsilon - end_epsilon) * progress)
        #epsilon = custom_two_phase_decay(ep_id, start_epsilon, end_epsilon, num_episodes)
        if ep_id % update_target_steps == 0:
            target_model.set_weights(online_model.get_weights())


    # Save the online model at the end of training
    save_model(online_model, save_path)

    return online_model, target_model