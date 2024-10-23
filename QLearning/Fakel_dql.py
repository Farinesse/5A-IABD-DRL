import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
from tqdm import tqdm
from environment.FarkelEnv import FarkleEnv


@tf.function(reduce_retracing=True)
def gradient_step(model, s, a, target, optimizer):
    with tf.GradientTape() as tape:
        s = tf.ensure_shape(s, [None])  # Assurer une forme constante
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


def epsilon_greedy_action(q_s, valid_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        q_s_valid = tf.gather(q_s, valid_actions)
        return valid_actions[tf.argmax(q_s_valid)]


class FarkleDQNAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.99, memory_size=1000,
                 batch_size=16, target_update=50):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)

        self.state_size = env.observation_space.shape[0]
        self.action_size = 128  # 2^7 pour représenter toutes les combinaisons possibles d'actions

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        valid_actions = self.env.get_valid_actions()
        valid_indices = np.where(valid_actions == 1)[0]
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        q_values = model_predict(self.model, state_tensor)
        action_index = epsilon_greedy_action(q_values, valid_indices, self.epsilon)
        return [int(b) for b in format(action_index, '07b')]

    def train(self, num_episodes):
        total_score = 0.0
        total_loss = 0.0
        steps = 0
        episode_rewards = []

        for episode in tqdm(range(num_episodes), desc="Training", unit="episode"):
            state, _ = self.env.reset()
            done = False
            episode_score = 0
            episode_loss = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                self.remember(state, action, reward, next_state, done)
                episode_score += reward

                if len(self.memory) >= self.batch_size:
                    loss = self.replay()
                    episode_loss += loss
                    total_loss += loss

                state = next_state
                steps += 1

                if steps % self.target_update == 0:
                    self.target_model.set_weights(self.model.get_weights())

            total_score += episode_score
            episode_rewards.append(episode_score)

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 10 == 0:
                avg_score = np.mean(episode_rewards[-10:])
                avg_loss = total_loss / 10
                sum_rewards = np.sum(episode_rewards[-10:])
                print(f"Episode {episode + 1}, Avg Score: {avg_score:.2f}, Sum Rewards: {sum_rewards:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
                total_loss = 0.0

        return self.model
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in minibatch:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)
            action_index = int(''.join(map(str, action)), 2)

            if done:
                target = reward
            else:
                next_q_values = model_predict(self.target_model, next_state_tensor)
                valid_actions = self.env.get_valid_actions()
                valid_indices = np.where(valid_actions == 1)[0]
                next_q_values_valid = tf.gather(next_q_values, valid_indices)
                target = reward + self.gamma * tf.reduce_max(next_q_values_valid)

            loss = gradient_step(self.model, state_tensor, action_index, target, self.optimizer)
            total_loss += loss

        return total_loss / self.batch_size


def train(episodes=1000):
    env = FarkleEnv()
    agent = FarkleDQNAgent(env)
    trained_model = agent.train(episodes)
    return agent, trained_model


def evaluate(agent, episodes=100):
    env = FarkleEnv()
    scores = []

    for episode in tqdm(range(episodes), desc="Evaluation", unit="episode"):
        state, _ = env.reset()
        done = False
        episode_score = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            episode_score += reward

        scores.append(episode_score)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Evaluation - Mean Score: {mean_score:.2f}, Std Dev: {std_score:.2f}")
    return mean_score, std_score


if __name__ == "__main__":
    trained_agent, trained_model = train(episodes=100)
    mean_score, std_score = evaluate(trained_agent)