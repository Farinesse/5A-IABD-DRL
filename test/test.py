import tensorflow as tf
import numpy as np
import time
from statistics import mean
from algos.DQN.ddqn import epsilon_greedy_action
from environment.FarkelEnv import FarkleDQNEnv


def model_predict(model, state):
    result = model(tf.expand_dims(state, 0))
    if isinstance(result, dict):
        result = next(iter(result.values()))
    return result.numpy()


def epsilon_greedy_action(q_values, mask, available_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(available_actions)
    q_values = q_values[0]
    masked_q_values = q_values * mask
    return int(tf.argmax(masked_q_values))


def play_with_dqn(env, model, random_agent=None, episodes=1):
    total_rewards = 0
    episode_scores = []
    episode_times = []
    total_time = 0

    print("\nDétails des épisodes :")
    print("----------------------")

    for episode in range(episodes):
        start_time = time.time()
        env.reset()
        episode_reward = 0
        nb_turns = 0

        while not env.is_game_over():
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            result = model(tf.expand_dims(s_tensor, 0))
            q_s = result['dense_4']

            a = epsilon_greedy_action(q_s.numpy(), mask_tensor, env.available_actions_ids(), 0.000001)

            if a not in env.available_actions_ids():
                print(f"Action invalide {a}, choix aléatoire à la place.")
                a = np.random.choice(env.available_actions_ids())

            prev_score = env.score()
            env.step(a)
            reward = env.score() - prev_score
            episode_reward += reward
            nb_turns += 1

            env.display()

        end_time = time.time()
        episode_time = end_time - start_time

        episode_scores.append(episode_reward)
        episode_times.append(episode_time)
        total_rewards += episode_reward
        total_time += episode_time

        print(f"\nÉpisode {episode + 1}/{episodes}:")
        print(f"  Score: {episode_reward}")
        print(f"  Temps: {episode_time:.2f} secondes")
        print(f"  Nombre de tours: {nb_turns}")

    # Statistiques finales
    print("\nStatistiques globales:")
    print("---------------------")
    print(f"Score moyen: {mean(episode_scores):.2f}")
    print(f"Temps moyen par épisode: {mean(episode_times):.2f} secondes")
    print(f"Meilleur score: {max(episode_scores)}")
    print(f"Pire score: {min(episode_scores)}")
    print(f"Temps total: {total_time:.2f} secondes")

    # Afficher l'évolution des scores
    print("\nÉvolution des scores:")
    for i, score in enumerate(episode_scores, 1):
        print(f"Épisode {i}: {score}")


if __name__ == "__main__":
    env = FarkleDQNEnv()
    model_path = "models/models/double_dqn_model_Farkel_test1"
    model = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")

    print("Test initial...")
    s = env.state_description()
    mask = env.action_mask()
    result = model(tf.expand_dims(tf.convert_to_tensor(s, dtype=tf.float32), 0))
    q_values = result['dense_4'].numpy()
    print("Shape des Q-values:", q_values.shape)
    print("Actions disponibles:", env.available_actions_ids())

    # Lancer le jeu avec statistiques détaillées
    play_with_dqn(env, model, random_agent=None, episodes=1000
                  )