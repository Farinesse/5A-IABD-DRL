# DRL Project

## Description
Ce projet est une interface de ligne de commande interactive pour l'expérimentation et l'évaluation d'algorithmes d'apprentissage par renforcement profond (Deep Reinforcement Learning) sur différents environnements.

# DRL Project

[... contenu précédent ...]

## Configuration des Chemins de Modèles
Les chemins des modèles sont configurables dans la classe DRLInterface. Vous pouvez modifier les chemins dans `self.model_paths` pour pointer vers vos propres modèles :

```python
model_paths = {
    "Farkle": {
        "Deep Q-Learning": r"votre/chemin/vers/modele/dqn.pkl",
        "REINFORCE": r"votre/chemin/vers/modele/reinforce.pkl",
        # ...
    },
    "TicTacToe": {
        "Deep Q-Learning": r"votre/chemin/vers/modele/tictactoe_dqn.pkl",
        # ...
    },
    # ... autres environnements
}

```

Pour chaque environnement et algorithme, vous pouvez spécifier le chemin vers votre fichier de modèle `.pkl`. Assurez-vous que les chemins sont absolus ou relatifs au répertoire d'exécution du script.


## Environnements Disponibles
- TicTacToe: Jeu du morpion classique
- Farkle: Jeu de dés
- LineWorld: Environnement de navigation linéaire
- GridWorld: Environnement de navigation en grille 2D

## Algorithmes Implémentés
- Deep Q-Learning (DQN)
- DQN with Experience Replay
- Double Deep Q-Learning
- Double DQN with Experience Replay
- REINFORCE
- REINFORCE with Mean Baseline
- REINFORCE with Critic Baseline
- PPO (Proximal Policy Optimization)
- MCTS (Monte Carlo Tree Search)
- Random Agent

## Configuration Requise
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Colorama
- Pyfiglet

## Installation
```bash
# Cloner le repository
git clone [url-du-repo]

# Installer les dépendances
pip install tensorflow numpy pandas matplotlib colorama pyfiglet
```

## Structure du Projet
```
project/
├── algos/
│   ├── model_based/
│   │   └── mtcs_utc.py
│   └── ...
├── environment/
│   ├── tictactoe.py
│   ├── FarkelEnv.py
│   ├── line_word.py
│   └── grid_word.py
├── GUI/
│   ├── Farkel_GUI.py
│   └── test.py
├── functions/
│   ├── outils.py
│   └── random.py
├── models/
│   ├── farkle/
│   ├── tictactoe/
│   ├── line/
│   └── grid/
└── main.py
```

## Utilisation
1. Lancer l'application :
```bash
python main.py
```

2. Menu Principal :
   - Jouer : Test interactif des environnements et agents
   - Entraîner : Visualisation des métriques d'entraînement
   - Tester : Évaluation des modèles entraînés
   - Quitter : Fermer l'application

### Mode Jouer
- Permet de jouer contre différents agents (Random, DQN, REINFORCE, etc.)
- Options de jeu humain vs agent ou agent vs agent
- Visualisation du déroulement des parties

### Mode Entraînement
- Visualisation des métriques d'entraînement à partir de fichiers CSV
- Graphiques de performance incluant :
  - Score moyen
  - Temps par épisode
  - Taux de victoire
  - Nombre moyen d'étapes
  - Temps moyen par étape

### Mode Test
- Évaluation des modèles entraînés
- Génération de métriques de performance
- Sauvegarde des résultats en CSV
- Visualisation graphique des performances

## Extensions des Fichiers
- `.pkl` : Modèles sauvegardés
- `.csv` : Fichiers de métriques
- `.png` : Visualisations générées

## Notes
- Les modèles sont sauvegardés automatiquement avec un identifiant unique
- Les métriques sont sauvegardées dans le même dossier que les modèles
- Certains algorithmes peuvent nécessiter un temps de calcul significatif

## Dépannage
Si vous rencontrez des erreurs :
1. Vérifiez que tous les chemins de modèles sont corrects
2. Assurez-vous que les dépendances sont correctement installées
3. Vérifiez les permissions d'accès aux dossiers de sauvegarde

## Contribution
Pour contribuer au projet :
1. Fork le repository
2. Créez une branche pour votre fonctionnalité
3. Soumettez une pull request
