import tensorflow as tf
import keras
import numpy as np
from collections import deque
import random
from tqdm import tqdm  # Ajout pour la barre de progression

from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

from environment.FarkelEnv import FarkleEnv


class DQN:
    def __init__(self, input_size, output_size):
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(input_size,)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(output_size, activation='linear')
        ])
        self.model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')


class FarkleDQNAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000,
                 batch_size=64, target_update=10):
        self.env = env
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps = 0

        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n

        self.q_network = DQN(input_size, output_size).model
        self.target_network = DQN(input_size, output_size).model
        self.target_network.set_weights(self.q_network.get_weights())

    def select_action(self, state):
        valid_actions = self.env.get_valid_actions()
        valid_indices = np.where(valid_actions == 1)[0]

        if len(valid_indices) == 0:
            print("Aucune action valide disponible, fin de l'épisode ou gérer l'erreur.")
            return 0  # Ou une autre action par défaut

        if random.random() < self.epsilon:
            return np.random.choice(valid_indices)

        state_tensor = np.expand_dims(state, axis=0)
        q_values = self.q_network.predict(state_tensor, verbose=0)[0]
        q_values[valid_actions == 0] = -np.inf
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)

        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        loss = self.q_network.fit(states, targets, batch_size=self.batch_size, verbose=0).history['loss'][0]
        return loss

        if self.steps % self.target_update == 0:
            self.target_network.set_weights(self.q_network.get_weights())


        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train(episodes=10):
    env = FarkleEnv()
    agent = FarkleDQNAgent(env)
    scores = []

    print("Entraînement de l'agent DQN...")
    for episode in tqdm(range(episodes), desc="Progression de l'entraînement"):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()

            state = next_state
            episode_reward += reward

        scores.append(episode_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(scores[-10:])
            print(f"Épisode {episode}, Récompense Moyenne : {avg_reward:.2f}, Epsilon : {agent.epsilon:.2f}, Perte : {loss:.4f}")

    return agent, scores


def evaluate(agent, episodes=1):
    env = FarkleEnv()
    scores = []

    print("Évaluation de l'agent DQN...")
    for episode in tqdm(range(episodes), desc="Progression de l'évaluation"):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            episode_reward += reward

        scores.append(episode_reward)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Évaluation - Score Moyen : {mean_score:.2f}, Écart-type : {std_score:.2f}")
    return mean_score, std_score


if __name__ == "__main__":
    trained_agent, training_scores = train(episodes=100)
    mean_score, std_score = evaluate(trained_agent)
    print(f"Évaluation - Score Moyen : {mean_score:.2f}, Écart-type : {std_score:.2f}")
