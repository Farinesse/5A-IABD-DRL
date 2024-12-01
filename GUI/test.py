import pickle
import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import losses


@tf.function(reduce_retracing=True)
def predict_func(model, s):
    s = tf.ensure_shape(s, [None])
    output = model(tf.expand_dims(s, 0))
    return output


def epsilon_greedy_action_bis(q_s: tf.Tensor, mask: tf.Tensor, available_actions: np.ndarray, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        masked_q_s = q_s * mask + (1.0 - mask) * tf.float32.min
        # Modification ici pour gérer correctement le tensor
        return int(tf.argmax(masked_q_s[0]).numpy())  # Ajout de [0] et .numpy()


def load_model_pkl(file_path):
    #file_path = r'C:\Users\farin\PycharmProjects\5A-IABD-DRL\environment\farkle5000_100000_ddqn_noreplay_3d00ef49.pkl'
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
            print(type(model))
            return model
    except Exception as e:
        print(f"Erreur : {e}")
        return None


def action_agent(env, model):
    s = env.state_description()
    s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

    # Obtenir le masque des actions valides
    mask = np.zeros(128, dtype=np.float32)
    valid_actions = env.get_valid_actions()
    mask = valid_actions
    mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

    # Prédire l'action
    q_s = predict_func(model, s_tensor)

    a = epsilon_greedy_action_bis(q_s.numpy(), mask_tensor, env.get_valid_actions(), 0.000001)
    print('state', env.state_description())
    print('action', env.decode_action_1(a))
    return env.decode_action_1(a)

