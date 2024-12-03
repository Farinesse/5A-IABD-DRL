import numpy as np
from tqdm import tqdm  # Pour afficher la progression


def tabular_q_learning(env_type,
                       alpha: float = 0.1,
                       epsilon: float = 0.1,
                       gamma: float = 0.999,
                       nb_iter: int = 100000):

    Q = {}

    env = env_type()

    for it in tqdm(range(nb_iter)):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()

            aa = env.available_actions_ids()

            if s not in Q:
                Q[s] = {a: 0.0 for a in aa}

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                q_s = [Q[s][a] for a in aa]
                best_a_index = np.argmax(q_s)
                a = aa[best_a_index]

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()

            if env.is_game_over():
                target = r
            else:
                if s_p not in Q:
                    Q[s_p] = {a_p: 0.0 for a_p in aa_p}
                q_s_p = [Q[s_p][a_p] for a_p in aa_p]
                max_a_p = np.max(q_s_p)
                target = r + gamma * max_a_p

            Q[s][a] = (1 - alpha) * Q[s][a] + alpha * target

    Pi = {}
    for s in Q.keys():
        best_a = None
        best_a_score = float('-inf')

        for a, a_score in Q[s].items():
            if best_a is None or best_a_score < a_score:
                best_a = a
                best_a_score = a_score

        Pi[s] = best_a

    return Pi, Q
