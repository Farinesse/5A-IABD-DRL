import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm
from collections import deque
import random

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
    s = tf.ensure_shape(s, [None])  # Assurer une forme constante
    return model(tf.expand_dims(s, 0))[0]

def epsilon_greedy_action(
        q_s: tf.Tensor,
        mask: tf.Tensor,
        available_actions: np.ndarray,
        epsilon: float
) -> int:
    """
    Choisit une action avec une politique epsilon-greedy en s'assurant de ne pas choisir d'actions invalides.
    """
    if np.random.rand() < epsilon:
        # Choisir une action aléatoire parmi les actions disponibles
        return np.random.choice(available_actions)
    else:
        # Appliquer le masque pour bloquer les actions invalides
        masked_q_s = q_s * mask + (1.0 - mask) * tf.float32.min
        action = int(tf.argmax(masked_q_s, axis=0))

        # S'assurer que l'action choisie est dans les actions valides
        if action not in available_actions:
            print(f"Action {action} invalide, prise aléatoire à la place.")
            action = np.random.choice(available_actions)
        return action

def deep_q_learning(model, target_model, env, num_episodes, gamma, alpha, start_epsilon, end_epsilon,
                    memory_size=5000, batch_size=32, update_target_steps=500, epsilon_decay=0.999):
    optimizer = keras.optimizers.SGD(learning_rate=alpha, weight_decay=1e-7)
    memory = deque(maxlen=memory_size)
    epsilon = start_epsilon
    total_score = 0.0
    total_loss = 0.0

    for ep_id in tqdm(range(num_episodes)):
        if ep_id % 100 == 0 and ep_id > 0:
            print(f"Mean Score: {total_score / 100}, Mean Loss: {total_loss / 100}, Epsilon: {epsilon}")
            total_score = 0.0
            total_loss = 0.0

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

            # Assurez-vous que l'action choisie est valide
            if a not in env.available_actions_ids():
                print(f"Action {a} invalide, prise aléatoire à la place.")
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
                        q_next = model_predict(target_model, next_state)
                        target = reward + gamma * tf.reduce_max(q_next*next_mask_tensor)

                    loss = gradient_step(model, state, action, target, optimizer)
                    total_loss += loss.numpy()

            s_tensor = s_prime_tensor

        total_score += env.score()
        epsilon = max(end_epsilon, epsilon * epsilon_decay)

        if ep_id % update_target_steps == 0:
            target_model.set_weights(model.get_weights())

    return model
