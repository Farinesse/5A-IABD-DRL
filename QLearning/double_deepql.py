import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm
from collections import deque
import random

# Fonction pour effectuer un pas de gradient
@tf.function
def gradient_step(model, s, a, target, optimizer):
    with tf.GradientTape() as tape:
        q_s_a = model(tf.expand_dims(s, 0))[0][a]
        loss = tf.square(q_s_a - target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def model_predict(model, s):
    return model(tf.expand_dims(s, 0))[0]

def epsilon_greedy_action(q_s, mask, available_actions, epsilon):
    """
    Choix de l'action avec une politique epsilon-greedy.
    :param q_s: Les valeurs Q pour l'état actuel.
    :param mask: Le masque des actions valides.
    :param available_actions: Les actions disponibles.
    :param epsilon: Le taux d'exploration (epsilon).
    :return: L'action choisie.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        # Utilisation du masque pour bloquer les actions invalides
        inverted_mask = tf.constant(1.0) - mask
        masked_q_s = q_s * mask + tf.float32.min * inverted_mask
        return int(tf.argmax(masked_q_s, axis=0))

def deep_q_learning(model, target_model, env, num_episodes, gamma, alpha, start_epsilon, end_epsilon,
                    memory_size=128, batch_size=32, update_target_steps=100, epsilon_decay=0.995):
    """
    Fonction Deep Q-Learning avec Experience Replay et Fixed Target Network.
    :param model: Le modèle DQN principal.
    :param target_model: Le modèle DQN cible.
    :param env: L'environnement de jeu.
    :param num_episodes: Nombre total d'épisodes.
    :param gamma: Taux de discount.
    :param alpha: Taux d'apprentissage.
    :param start_epsilon: Valeur initiale de epsilon pour l'exploration.
    :param end_epsilon: Valeur finale de epsilon.
    :param memory_size: Taille de la mémoire de répétition d'expérience.
    :param batch_size: Taille du lot pour l'entraînement.
    :param update_target_steps: Nombre d'épisodes avant la mise à jour du modèle cible.
    :param epsilon_decay: Facteur de décroissance de epsilon.
    :return: Le modèle DQN entraîné.
    """
    optimizer = keras.optimizers.Adam(learning_rate=alpha)
    memory = deque(maxlen=memory_size)  # Replay Memory
    epsilon = start_epsilon  # Valeur initiale de epsilon

    total_score = 0.0

    for ep_id in tqdm(range(num_episodes)):
        if ep_id % 100 == 0 and ep_id > 0:
            print(f"Mean Score: {total_score / 100}")
            total_score = 0.0

        env.reset()
        if env.is_game_over():
            continue

        s = env.state_description()
        s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

        while not env.is_game_over():
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_s = model_predict(model, s_tensor)
            a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), epsilon)

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)
            done = env.is_game_over()

            # Ajouter la transition à la mémoire
            memory.append((s_tensor, a, r, s_prime_tensor, done))

            if len(memory) >= batch_size:
                # Échantillonner un mini-lot de transitions
                minibatch = random.sample(memory, batch_size)

                for state, action, reward, next_state, done in minibatch:
                    if done:
                        target = reward
                    else:
                        next_mask = env.action_mask()
                        next_mask_tensor = tf.convert_to_tensor(next_mask, dtype=tf.float32)

                        # Double Q-Learning : utiliser le modèle principal pour choisir l'action, et le modèle cible pour évaluer
                        q_next = model_predict(target_model, next_state)
                        q_next_main = model_predict(model, next_state)
                        best_action = tf.argmax(q_next_main).numpy()
                        target = reward + gamma * q_next[best_action]

                    # Mettre à jour le modèle principal
                    gradient_step(model, state, action, target, optimizer)

            # Passer à l'état suivant
            s_tensor = s_prime_tensor

        total_score += env.score()

        # Diminution de epsilon après chaque épisode
        epsilon = max(end_epsilon, epsilon * epsilon_decay)

        # Mettre à jour le modèle cible tous les "update_target_steps" épisodes
        if ep_id % update_target_steps == 0:
            target_model.set_weights(model.get_weights())

    return model
