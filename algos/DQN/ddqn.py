import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm
from functions.outils import log_metrics_to_dataframe, play_with_dqn, epsilon_greedy_action


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


@tf.function(reduce_retracing=True)
def model_predict(
        model,
        s
):
    s = tf.ensure_shape(s, [None])  # Ensure constant shape
    return model(tf.expand_dims(s, 0))[0]


def save_model(
        model,
        file_path,
        save_format="tf"
):
    """
    Sauvegarde le modèle dans un fichier.

    :param model: Le modèle à sauvegarder
    :param file_path: Le chemin du fichier où sauvegarder le modèle
    :param save_format: Le format de sauvegarde ('tf' pour TensorFlow ou 'h5' pour HDF5)
    """
    try:
        model.save(file_path, save_format=save_format)
        print(f"Modèle sauvegardé avec succès dans {file_path} au format {save_format}")
    except ImportError as e:
        print(f"Erreur d'importation (vérifiez TensorFlow et h5py) : {e}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")


def double_dqn_no_replay(
        online_model,
        target_model,
        env,
        num_episodes,
        gamma,
        alpha,
        start_epsilon,
        end_epsilon,
        update_target_steps=1000,
        save_path='models/double_dqn_model_Farkel_test1',
        input_dim=12
):
    optimizer = keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=0.99,  # Ajout de momentum pour une convergence plus rapide
        nesterov=True,  # Utilisation de Nesterov momentum pour une meilleure performance
        weight_decay=1e-4 # Ajout de régularisation L2 pour éviter le surapprentissage
    )

    # optimizer = keras.optimizers.Adam(learning_rate=alpha)

    epsilon = start_epsilon
    total_loss = 0.0
    interval = 1000
    results_df = None


    for ep_id in tqdm(range(num_episodes)):
        if ep_id % interval == 0 and ep_id > 0:

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

            loss = gradient_step(online_model, s_tensor, a, target, optimizer, input_dim)
            total_loss += loss.numpy()

        progress = ep_id / num_episodes
        epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        if ep_id % update_target_steps == 0:
            target_model.set_weights(online_model.get_weights())

    save_model(online_model, save_path, save_format="h5")

    results_df.to_csv(f'{save_path}_metrics.csv', index=False)

    return online_model, target_model
