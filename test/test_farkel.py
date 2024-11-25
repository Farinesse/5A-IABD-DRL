import numpy as np

from environment.FarkelEnv import FarkleEnv


class FarkleDQNEnv(FarkleEnv):
    def __init__(self, num_players=2, target_score=5000):
        super().__init__(num_players, target_score)
        self.action_space_size = 128  # 2^7 possibilités (6 dés + stop action)
        self.last_player0_score = 0 # Pour suivre le score du joueur 0
        self.last_player1_score = 0 # Pour suivre le score du joueur 1

    def available_actions_ids(self) -> np.ndarray:
        """
        Retourne les indices des actions valides.
        """
        valid_mask = self.get_valid_actions()
        return np.where(valid_mask == 1)[0]

    def action_mask(self) -> np.ndarray:
        """Masque binaire pour les actions valides."""
        mask = np.zeros(128, dtype=np.float32)
        valid_actions = self.available_actions_ids()
        mask[valid_actions] = 1.0
        return mask

    def decode_action(self, action_id):
        """Décode l'ID d'action en une liste d'actions binaires."""
        binary = format(action_id, '07b')  # Toujours 7 bits pour 6 dés + action stop
        return [int(b) for b in binary]

    def reset(self, seed=None):
        """Réinitialise l'environnement et les scores de suivi."""
        observation, info = super().reset(seed)
        self.last_player0_score = 0
        self.last_player1_score = 0
        return observation, info

    def step(self, action_id):
        """Exécute une action et fait jouer le joueur aléatoire si nécessaire."""
        # Sauvegarde des scores avant l'action
        self.last_player0_score = self.scores[0]
        self.last_player1_score = self.scores[1]

        # Tour du joueur 0 (agent)
        action = self.decode_action(action_id)
        observation, reward, done, truncated, info = super().step(action)

        # Si le jeu n'est pas terminé et c'est au tour du joueur aléatoire
        if not done and self.current_player == 1:
            # Faire jouer le joueur aléatoire
            random_initial_score = self.scores[1]
            self.play_random_turn()
            random_score_gained = self.scores[1] - random_initial_score

            # Ajuster la récompense en fonction de la performance du joueur aléatoire
            if random_score_gained > 0:
                reward = reward - (random_score_gained/self.target_score)  # Pénalité proportionnelle
            else:
                reward = reward + 0.1  # Petit bonus si l'adversaire n'a pas marqué

            # Vérifier si le jeu est terminé après le tour du joueur aléatoire
            if self.game_over:
                if self.scores[0] >= self.target_score:
                    reward += 1.0  # Bonus pour la victoire
                elif self.scores[1] >= self.target_score:
                    reward -= 1.0  # Pénalité pour la défaite
                done = True

            # Mettre à jour l'observation
            observation = self.get_observation()

        return observation, reward, done, truncated, info

    def score(self) -> float:
        """
        Calcule le score normalisé pour l'apprentissage.
        """
        # Si le jeu est terminé
        if self.game_over:
            if self.scores[0] >= self.target_score:
                return 1.0
            elif self.scores[1] >= self.target_score:
                return -1.0
            else:
                return 0.0

        # Score normalisé basé sur la différence relative des scores
        score_diff = (self.scores[0] - self.scores[1]) / self.target_score
        return np.clip(score_diff, -1.0, 1.0)

    def get_normalized_scores(self):
        """
        Retourne les scores normalisés des deux joueurs.
        """
        return [score / self.target_score for score in self.scores]

    def get_score_change(self):
        """
        Calcule le changement de score pour les deux joueurs depuis la dernière action.
        """
        player0_change = self.scores[0] - self.last_player0_score
        player1_change = self.scores[1] - self.last_player1_score
        return player0_change, player1_change

    def display(self):
        """
        Affiche l'état actuel du jeu avec des informations supplémentaires.
        """
        print(f"Joueur actuel: {'Agent' if self.current_player == 0 else 'Random'}")
        print(f"Score de la manche: {self.round_score}")
        print(f"Dés restants: {self.remaining_dice}")
        print(f"Dés actuels: {self.dice_roll}")
        print(f"Scores - Agent: {self.scores[0]}, Random: {self.scores[1]}")
        normalized_scores = self.get_normalized_scores()
        print(f"Scores normalisés - Agent: {normalized_scores[0]:.2f}, Random: {normalized_scores[1]:.2f}")
        print(f"Jeu terminé: {self.game_over}")


if __name__ == "__main__":
    # Test de l'environnement
    env = FarkleDQNEnv(target_score=2000)

    # Test sur quelques épisodes
    for episode in range(3):
        observation, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Action aléatoire pour le test
            valid_actions = env.available_actions_ids()
            action = np.random.choice(valid_actions)

            observation, reward, done, _, info = env.step(action)
            total_reward += reward

            env.display()
            print(f"Reward: {reward:.2f}")
            print("-" * 50)

        print(f"Episode {episode + 1} terminé, Reward total: {total_reward:.2f}\n")