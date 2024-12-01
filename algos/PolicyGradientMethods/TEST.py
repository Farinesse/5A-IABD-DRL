import tensorflow as tf
import numpy as np
import os


def load_reinforce_model(model_path):
    """
    Charge un modèle sauvegardé depuis le dossier checkpoints_actor
    """
    try:
        # S'assurer que le chemin pointe vers le dossier checkpoints_actor
        full_path = os.path.join('checkpoints_actor', model_path) if 'checkpoints_actor' not in model_path else model_path
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Le modèle {full_path} n'existe pas")

        model = tf.keras.models.load_model(full_path)
        print(f"Modèle chargé avec succès depuis: {full_path}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        return None


def play_with_reinforce(env, model_path, episodes=1, display=True):
    """
    Joue des parties avec un modèle REINFORCE sauvegardé

    Args:
        env: L'environnement de jeu
        model_path: Chemin vers le modèle sauvegardé
        episodes: Nombre d'épisodes à jouer
        display: Afficher les détails de la partie

    Returns:
        float: Score moyen sur tous les épisodes
    """
    # Charger le modèle
    model = load_reinforce_model(model_path)
    if model is None:
        print("Impossible de continuer sans modèle valide")
        return 0

    total_rewards = 0
    wins = 0
    game_lengths = []

    print(f"\nDébut des tests sur {episodes} épisodes avec le modèle: {model_path}")

    for episode in range(episodes):
        env.reset()
        done = False
        episode_reward = 0
        steps = 0

        if display:
            print(f"\n{'=' * 50}")
            print(f"Épisode {episode + 1}/{episodes}")
            print(f"{'=' * 50}")

        while not done:
            state = env.state_description()
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            valid_actions = env.available_actions_ids()

            if len(valid_actions) == 0:
                if display:
                    print("Aucune action valide disponible!")
                break

            # Obtenir les probabilités d'action
            probs = model(tf.expand_dims(state_tensor, 0), training=False)[0]
            mask = np.ones_like(probs.numpy()) * float('-inf')
            mask[valid_actions] = 0
            masked_probs = tf.nn.softmax(probs + mask).numpy()

            # Sélectionner l'action la plus probable parmi les actions valides
            action = valid_actions[np.argmax(masked_probs[valid_actions])]

            # Exécuter l'action
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            episode_reward += reward
            done = env.is_game_over()
            steps += 1

            if display:
                print("\nÉtat actuel:")
                env.display()
                print(f"Action choisie: {action}")
                print(f"Récompense: {reward}")
                print(f"Score actuel: {env.score()}")

        total_rewards += episode_reward
        game_lengths.append(steps)
        if episode_reward > 0:
            wins += 1

        if display:
            print(f"\nÉpisode {episode + 1} terminé")
            print(f"Score final: {env.score()}")
            print(f"Nombre de coups: {steps}")
            print(f"Victoire: {'Oui' if episode_reward > 0 else 'Non'}")

    # Statistiques finales
    mean_score = total_rewards / episodes
    win_rate = (wins / episodes) * 100
    avg_length = sum(game_lengths) / len(game_lengths)

    print(f"\nRésultats sur {episodes} parties:")
    print(f"{'=' * 50}")
    print(f"Score moyen: {mean_score:.2f}")
    print(f"Taux de victoire: {win_rate:.2f}%")
    print(f"Longueur moyenne des parties: {avg_length:.1f} coups")

    return mean_score


# Exemple d'utilisation:
if __name__ == "__main__":
    from environment.FarkelEnv import FarkleDQNEnv

    env = FarkleDQNEnv(target_score=2000)
    model_path = r"/algos/PolicyGradientMethods/checkpoints_actor\reinforce_policy_100000.h5"  # ou un autre checkpoint

    # Jouer 10 parties avec affichage
    score = play_with_reinforce(env, model_path, episodes=1000, display=False)