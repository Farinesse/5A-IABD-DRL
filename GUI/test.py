import numpy as np
import tensorflow as tf
from tensorflow.python.keras import losses
from functions.outils import epsilon_greedy_action
from tensorflow.keras.models import load_model as tf_load_model

def predict_func(model, state):
    # Sélectionner seulement les 11 premières caractéristiques
    state = state[:11]
    s = tf.ensure_shape(state, (None,))
    return model(tf.expand_dims(s, 0))[0]


def load_model(model_path, save_format="tf"):
    """
    Charge un modèle à partir d'un fichier donné.

    :param model_path: Le chemin du fichier contenant le modèle
    :param save_format: Le format de sauvegarde utilisé ("tf" ou "h5")
    :return: Le modèle chargé
    """
    try:
        # Charger le modèle
        model = tf_load_model(model_path, custom_objects=None, compile=True)
        print(f"Modèle chargé avec succès à partir de {model_path} au format {save_format}")
        return model
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

    # Vérifier la validité de l'action
    if a not in env.get_valid_actions():
        print(f"Action {a} invalide, prise aléatoire à la place.")
        a = np.random.choice(env.available_actions_ids())

    return env.decode_action_1(a)  # Changé de decode_action_1 à decode_action