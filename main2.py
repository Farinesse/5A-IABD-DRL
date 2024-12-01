import tensorflow as tf
import numpy as np

def epsilon_greedy_action_bis(q_s: tf.Tensor, mask: tf.Tensor, available_actions: np.ndarray, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        masked_q_s = q_s * mask + (1.0 - mask) * tf.float32.min
        # Modification ici pour gérer correctement le tensor
        return int(tf.argmax(masked_q_s[0]).numpy())  # Ajout de [0] et .numpy()

# Exemple de test avec des valeurs fictives
q_s = tf.convert_to_tensor([[0.5, 1.2, -0.3, 0.8, 1.1, 0.2, 1.4, -0.2, 0.9]], dtype=tf.float32)  # Q-values pour chaque action
mask = tf.convert_to_tensor([1, 1, 1, 1, 0, 1, 0, 1, 1], dtype=tf.float32)  # Masque où 1 indique une action valide
available_actions = np.array([0, 1, 2, 3, 5, 7, 8])  # Actions valides (indices)
epsilon = 0.1  # Taux d'exploration

# Appel à la fonction
chosen_action = epsilon_greedy_action_bis(q_s, mask, available_actions, epsilon)

print(f"Action choisie : {chosen_action}")
