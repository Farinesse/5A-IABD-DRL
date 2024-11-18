import numpy as np
from tqdm import tqdm  # Pour afficher la progression


def tabular_q_learning(env_type,
                       alpha: float = 0.1,
                       epsilon: float = 0.1,
                       gamma: float = 0.999,
                       nb_iter: int = 100000):
    """
    Implémentation de l'algorithme Tabular Q-Learning pour un environnement générique.
    :param env_type: Le type d'environnement (doit implémenter GameEnv)
    :param alpha: Taux d'apprentissage
    :param epsilon: Taux d'exploration pour epsilon-greedy
    :param gamma: Facteur de discount
    :param nb_iter: Nombre d'itérations d'entraînement
    :return: La politique Pi apprise et le modèle Q (la table Q apprise)
    """
    Q = {}  # Table Q

    env = env_type()  # Initialisation de l'environnement

    for it in tqdm(range(nb_iter)):  # Boucle sur le nombre d'itérations
        env.reset()

        while not env.is_game_over():
            s = env.state_id()  # Identification de l'état actuel

            aa = env.available_actions_ids()  # Actions possibles

            # Si l'état n'est pas déjà dans Q, on initialise la table Q pour cet état à 0
            if s not in Q:
                Q[s] = {a: 0.0 for a in aa}

            # Choix de l'action selon la stratégie epsilon greedy
            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                q_s = [Q[s][a] for a in aa]
                best_a_index = np.argmax(q_s)
                a = aa[best_a_index]

            prev_score = env.score()  # Récupérer le score précédent
            env.step(a)  # Exécuter l'action
            r = env.score() - prev_score  # Calculer la récompense immédiate

            s_p = env.state_id()  # Identification du nouvel état
            aa_p = env.available_actions_ids()  # Nouvelles actions possibles

            # Calcul de la cible pour la mise à jour de Q
            if env.is_game_over():
                target = r
            else:
                if s_p not in Q:
                    Q[s_p] = {a_p: 0.0 for a_p in aa_p}
                q_s_p = [Q[s_p][a_p] for a_p in aa_p]
                max_a_p = np.max(q_s_p)
                target = r + gamma * max_a_p

            # Mise à jour de la valeur Q
            Q[s][a] = (1 - alpha) * Q[s][a] + alpha * target

    # Construction de la politique Pi à partir de la table Q apprise
    Pi = {}
    for s in Q.keys():
        best_a = None
        best_a_score = float('-inf')

        for a, a_score in Q[s].items():
            if best_a is None or best_a_score < a_score:
                best_a = a
                best_a_score = a_score

        Pi[s] = best_a

    return Pi, Q  # Retourne la politique Pi et la table Q comme modèle
