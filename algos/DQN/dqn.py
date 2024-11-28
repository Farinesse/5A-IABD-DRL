
import random
import keras
import numpy as np
import tensorflow as tf
from keras.src import regularizers
from tqdm import tqdm

from environment.tictactoe import TicTacToe
from functions.outils import log_metrics_to_dataframe, play_with_dqn, epsilon_greedy_action


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


@tf.function
def model_predict(model, s):
    """Prédiction des Q-valeurs pour un état donné."""
    s = tf.ensure_shape(s, (None,))
    return model(tf.expand_dims(s, 0))[0]


def epsilon_greedy_action(q_s: tf.Tensor,mask: tf.Tensor,available_actions: np.ndarray,epsilon: float) -> int:

    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else :
        inverted_mask = tf.constant(1.0) - mask
        masked_q_s = q_s * mask + (1e-8) * inverted_mask
        return int(tf.argmax(masked_q_s, axis=0))


def save_model(model, file_path, save_format="tf"):
    """Sauvegarde du modèle."""
    try:
        model.save(file_path, save_format=save_format)
        print(f"Modèle sauvegardé avec succès dans {file_path} au format {save_format}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")


def soft_update_target_network(model, target_model, tau=0.01):
    """Mise à jour progressive des poids du modèle cible."""
    target_weights = target_model.get_weights()
    model_weights = model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = tau * model_weights[i] + (1 - tau) * target_weights[i]
    target_model.set_weights(target_weights)


def dqn_no_replay(model, target_model, env, num_episodes, gamma, alpha, start_epsilon, end_epsilon, update_frequency, save_path, input_dim):
    """Entraînement d'un agent DQN sans mémoire de rejouabilité."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.00001,  # Learning rate plus faible
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
        if ep_id % 500 == 0 and ep_id > 0:
            results_df = dqn_log_metrics_to_dataframe(
                function=play_with_dqn,
                model=model,
                predict_func=model_predict,
                env=env,
                episode_index=ep_id,
                games=100,
                dataframe=results_df
            )
            save_model(model, f"{save_path}_episode_{ep_id}.h5", save_format="h5")
            print(f"Épisode: {ep_id}, Perte moyenne: {total_loss / 500}, Epsilon: {epsilon}")
            total_loss = 0.0

    save_model(model, f"{save_path}_finale.h5", save_format="h5")
    if results_df is not None:
        results_df.to_csv(f"{save_path}_metrics.csv", index=False)
    return model, target_model


if __name__ == "__main__":
    from environment.FarkelEnv import FarkleDQNEnv

    # Configuration
    env = FarkleDQNEnv(num_players=2, target_score=5000)
    l2_reg = 0.01  # Force de la régularisation L2
    dropout_rate = 0.2  # Taux de dropout pour réduire l'overfitting

    # Création du modèle
    model = keras.Sequential([keras.layers.InputLayer(input_shape=(12,)),keras.layers.Normalization(),  # Ajout d'une couche de normalisation

        # Couche d'entrée
        keras.layers.Dense(128,activation='relu',kernel_regularizer=regularizers.L2(l2_reg),kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),

        # Couches cachées avec taille croissante
        keras.layers.Dense(256,activation='relu',kernel_regularizer=regularizers.L2(l2_reg),kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(512,activation='relu',kernel_regularizer=regularizers.L2(l2_reg),kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),

        # Couches cachées avec taille décroissante
        keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L2(l2_reg),kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(l2_reg), kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(128) ,
        keras.layers.LayerNormalization()

    ])

    # Entraînement
    model = dqn_no_replay(
        model=model,
        target_model=model,
        env=env,
        num_episodes=20000,
        gamma=0.99,
        alpha=0.00001,
        start_epsilon=1.0,
        end_epsilon=0.001,
        save_path ='models/dqn_replay/dqn_replay_model_farkel_5000_tests/20000-0.99_0-00001-500-tests/dqn_replay_model_farkel_test_10000_0-99_0-0001_1-0_0-01_64_32_100_128relu12dim_512relu_256relu_256relu_128',
        input_dim = 12,
        update_frequency=500
    )
