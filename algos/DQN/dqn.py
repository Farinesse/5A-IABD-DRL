import secrets
import keras
import tensorflow as tf
from tqdm import tqdm
from functions.outils import (
    log_metrics_to_dataframe,
    play_with_dqn,
    epsilon_greedy_action,
    plot_csv_data,
    save_model,
    dqn_model_predict as model_predict, save_files
)


@tf.function
def gradient_step(model, s, a, target, optimizer, input_dim):
    """Un pas de gradient pour entraîner le modèle."""
    with tf.GradientTape() as tape:
        s = tf.ensure_shape(s, (input_dim,))
        a = tf.cast(a, dtype=tf.int32)
        q_s_a = model(tf.expand_dims(s, 0))[0][a]
        loss = tf.square(q_s_a - target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def soft_update_target_network(model, target_model, tau=0.01):
    """Mise à jour progressive des poids du modèle cible."""
    target_weights = target_model.get_weights()
    model_weights = model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = tau * model_weights[i] + (1 - tau) * target_weights[i]
    target_model.set_weights(target_weights)


def dqn_no_replay(
        model,
        target_model,
        env,
        num_episodes,
        gamma,
        alpha,
        start_epsilon,
        end_epsilon,
        update_frequency,
        save_path = None,
        input_dim = None,
        interval = 10
):
    """Entraînement d'un agent DQN sans mémoire de rejouabilité."""
    optimizer = keras.optimizers.Adam(
        learning_rate=alpha,  # Learning rate plus faible
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    epsilon = start_epsilon
    total_loss = 0.0
    results_df = None
    steps = 0  # Compteur global des étapes

    target_model.set_weights(model.get_weights())  # Initialisation du modèle cible

    for ep_id in tqdm(range(num_episodes)):
        env.reset()
        if env.is_game_over():
            continue

        while not env.is_game_over():
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_values = model_predict(model, s_tensor)
            available_actions = env.available_actions_ids()

            action = epsilon_greedy_action(q_values, mask_tensor, available_actions, epsilon)

            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score

            if env.is_game_over():
                target = reward
            else:
                s_prime = env.state_description()
                s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)
                next_mask = env.action_mask()
                next_mask_tensor = tf.convert_to_tensor(next_mask, dtype=tf.float32)
                q_next = model_predict(target_model, s_prime_tensor)
                target = reward + gamma * tf.reduce_max(q_next * next_mask_tensor)

            loss = gradient_step(model, s_tensor, action, target, optimizer, input_dim)
            total_loss += loss.numpy()
            steps += 1

            if steps % update_frequency == 0:
                soft_update_target_network(model, target_model)

        # Mise à jour d'epsilon
        epsilon = max(end_epsilon, start_epsilon * (1 - ep_id / num_episodes))

        # Sauvegarde périodique et affichage des métriques
        if (ep_id + 1) % interval == 0 and ep_id > 0:
            results_df = log_metrics_to_dataframe(
                function=play_with_dqn,
                model=model,
                predict_func=model_predict,
                env=env,
                episode_index=ep_id,
                games=100,
                dataframe=results_df
            )
            print(f"Épisode: {ep_id}, Perte moyenne: {total_loss / 500}, Epsilon: {epsilon}")
            total_loss = 0.0

    if save_path is not None:
        save_files(
            model,
            "DQN NO REPLAY",
            results_df,
            env,
            num_episodes,
            gamma,
            alpha,
            start_epsilon,
            end_epsilon,
            update_frequency,
            optimizer,
            save_path=save_path
        )

    return model, target_model

