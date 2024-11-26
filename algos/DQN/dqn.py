
import random
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from environment.tictactoe import TicTacToe
from functions.outils import dqn_log_metrics_to_dataframe, play_with_dqn, epsilon_greedy_action


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
        s = tf.ensure_shape(s, [input_dim])
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
    s = tf.ensure_shape(s, [None])
    return model(tf.expand_dims(s, 0))[0]


def debug_action_selection(
        env,
        model,
        epsilon
):
    s = env.state_description()
    s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
    mask = env.action_mask()
    mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
    available_actions = env.available_actions_ids()

    q_s = model_predict(model, s_tensor)

    print("État actuel:", s)
    print("Masque d'action:", mask)
    print("Actions disponibles:", available_actions)
    print("Valeurs Q:", q_s.numpy())

    action = epsilon_greedy_action(q_s, mask_tensor, available_actions, epsilon)

    if action == -1:
        print("Aucune action disponible. Le jeu est probablement terminé.")
        return action

    print("Action choisie:", action)
    assert action in available_actions, f"Action {action} non valide!"
    return action


def save_model(
        model,
        file_path,
        save_format="tf"
):
    try:
        model.save(file_path, save_format=save_format)
        print(f"Modèle sauvegardé avec succès dans {file_path} au format {save_format}")
    except ImportError as e:
        print(f"Erreur d'importation (vérifiez TensorFlow et h5py) : {e}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")

def soft_update_target_network(model, target_model, tau=0.01):
    """Mise à jour progressive du target network"""
    target_weights = target_model.get_weights()
    weights = model.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]

    target_model.set_weights(target_weights)


def dqn_no_replay(
        model,
        target_model,  # Ajout du target model
        env,
        num_episodes,
        gamma,
        alpha,
        start_epsilon,
        end_epsilon,
        tau=0.01,
        update_frequency=100,

        save_path='dqn_model_farkel.h5',
        input_dim=12
):
    optimizer = keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=0.99,
        nesterov=True,
        weight_decay=1e-4
    )

    epsilon = start_epsilon
    total_loss = 0.0
    interval = 100
    results_df = None
    steps = 0  # Compteur pour update_target_steps

    # Initialiser target_model avec les poids de model
    target_model.set_weights(model.get_weights())

    for ep_id in tqdm(range(num_episodes)):
        if ep_id % interval == 0 and ep_id > 0:
            results_df = dqn_log_metrics_to_dataframe(
                function=play_with_dqn,
                model=model,
                predict_func=model_predict,
                env=env,
                episode_index=ep_id,
                games=100,
                dataframe=results_df
            )
            save_model(model, f"{save_path}_{ep_id}.h5", save_format="h5")
            print(f"Mean Loss: {total_loss / interval}, Epsilon: {epsilon}")
            total_loss = 0.0

        env.reset()
        if env.is_game_over():
            continue

        while not env.is_game_over():
            # État actuel
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            # Sélection de l'action avec le modèle principal
            q_s = model_predict(model, s_tensor)
            a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), epsilon)

            if a not in env.available_actions_ids():
                print(f"Action {a} invalide, prise aléatoire à la place.")
                a = np.random.choice(env.available_actions_ids())

            # Exécution de l'action
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            # Apprentissage immédiat avec target network
            if env.is_game_over():
                target = r
            else:
                s_prime = env.state_description()
                s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)
                next_mask = env.action_mask()
                next_mask_tensor = tf.convert_to_tensor(next_mask, dtype=tf.float32)

                # Utiliser target_model pour la prédiction des Q-values futures
                q_next = model_predict(target_model, s_prime_tensor)
                target = r + gamma * tf.reduce_max(q_next * next_mask_tensor)

            # Mise à jour du réseau principal
            loss = gradient_step(model, s_tensor, a, target, optimizer, input_dim)
            total_loss += loss.numpy()

            steps += 1

            # Mise à jour périodique du target network

            if steps % update_frequency == 0:
                soft_update_target_network(model, target_model, tau)

        # Mise à jour d'epsilon
        progress = ep_id / num_episodes
        epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

    # Sauvegarde finale
    save_model(model, f'{save_path}_finale.h5', save_format="h5")
    if results_df is not None:
        results_df.to_csv(f'{save_path}_metrics.csv', index=False)

    return model, target_model


if __name__ == "__main__":
    from environment.FarkelEnv import FarkleDQNEnv

    # Configuration
    env = FarkleDQNEnv(num_players=2, target_score=5000)

    # Création du modèle
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_dim=12),  # 3 + num_players + 6 + 1
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128)  # Nombre d'actions possibles dans Farkle
    ])

    # Entraînement
    model = dqn_no_replay(
        model=model,
        target_model=model,
        env=env,
        num_episodes=50000,
        gamma=0.995,
        alpha=0.0001,
        start_epsilon=1.0,
        end_epsilon=0.001,
        save_path ='../models/models/dqn_replay/dqn_replay_model_farkel_5000_tests/dqn_replay_model_farkel_test_10000_0-99_0-0001_1-0_0-01_64_32_100_128relu12dim_512relu_256relu_256relu_128',
        input_dim = 12
    )
