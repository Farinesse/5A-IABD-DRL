import random
import secrets
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import deque
from functions.outils import (
    log_metrics_to_dataframe,
    play_with_dqn,
    epsilon_greedy_action,
    plot_csv_data,
    save_model,
    dqn_model_predict as model_predict, save_files
)

@tf.function(reduce_retracing=True)
def gradient_step(
        model,
        s,
        a,
        target,
        optimizer,
        input_dim
):
    with tf.GradientTape() as tape:
        s = tf.ensure_shape(s, [input_dim])  # Dynamically use input_dim
        a = tf.cast(a, dtype=tf.int32)
        q_s_a = model(tf.expand_dims(s, 0))[0][a]
        loss = tf.square(q_s_a - target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def double_dqn_with_replay(
        online_model,
        target_model,
        env,
        num_episodes,
        gamma,
        alpha,
        start_epsilon,
        end_epsilon,
        memory_size=512,
        batch_size=32,
        update_target_steps=1000,
        save_path=None,
        input_dim=None,
        interval = 10
):
    optimizer = keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=0.99,  # Ajout de momentum pour une convergence plus rapide
        nesterov=True,  # Utilisation de Nesterov momentum pour une meilleure performance
        weight_decay=1e-4 # Ajout de régularisation L2 pour éviter le surapprentissage
    )

    # optimizer = keras.optimizers.Adam(learning_rate=alpha)

    memory = deque(maxlen=memory_size)
    epsilon = start_epsilon
    total_loss = 0.0
    results_df = None

    for ep_id in tqdm(range(num_episodes)):
        if (ep_id + 1) % interval == 0 and ep_id > 0:
            results_df = log_metrics_to_dataframe(
                function = play_with_dqn,
                model = online_model,
                predict_func = model_predict,
                env = env,
                episode_index = ep_id,
                games = 1000,
                dataframe = results_df
            )

            print(f"Mean Loss: {total_loss / interval}, Epsilon: {epsilon}")
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

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)

            memory.append((s_tensor, a, r, s_prime_tensor, env.is_game_over()))

            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)

                for state, action, reward, next_state, done in minibatch:
                    if done:
                        target = reward
                    else:
                        next_mask = env.action_mask()
                        next_mask_tensor = tf.convert_to_tensor(next_mask, dtype=tf.float32)

                        # Double Q-learning update
                        q_next_online = model_predict(online_model, next_state)
                        q_next_target = model_predict(target_model, next_state)
                        best_action = tf.argmax(q_next_online * next_mask_tensor)
                        target = reward + gamma * q_next_target[best_action]

                    loss = gradient_step(online_model, state, action, target, optimizer, input_dim)
                    total_loss += loss.numpy()

        progress = ep_id / num_episodes
        epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        if ep_id % update_target_steps == 0:
            target_model.set_weights(online_model.get_weights())


    if save_path is not None:
        save_files(
            online_model,
            "DDQN EXP REPLAY",
            results_df,
            env,
            num_episodes,
            gamma,
            alpha,
            start_epsilon,
            end_epsilon,
            update_target_steps,
            optimizer,
            save_path=save_path
        )

    return online_model, target_model