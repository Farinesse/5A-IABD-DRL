import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import losses

from functions.outils import epsilon_greedy_action, load_model_pkl
from algos.DQN.ddqn import model_predict as predict_func


def create_model():
    """
    Crée le modèle DQN avec l'architecture spécifiée:
    512relu12dim_256relu_dropout0.2_256relu_dropout0.2_128
    """
    model = keras.Sequential([
        # Première couche - 512 neurones avec input_shape=12
        keras.layers.Dense(512, activation='relu', input_shape=(12,)),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(128)
    ])
    return model

def load_model(model_path):

    try:

        loaded_model = load_model_pkl(model_path)
        print(f"Modèle chargé avec succès à partir de {model_path} au format")
        return loaded_model
    except ValueError as ve:
        print(f"Erreur lors du chargement du modèle : {ve}")
    except ImportError as ie:
        print(f"Erreur d'importation (vérifiez TensorFlow et h5py) : {ie}")
    except Exception as e:
        print(f"Erreur inattendue lors du chargement du modèle : {e}")


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
    a = epsilon_greedy_action(q_s.numpy(), mask_tensor, env.get_valid_actions(), 0.000001)

    return env.decode_action_1(a)  # Changé de decode_action_1 à decode_action```