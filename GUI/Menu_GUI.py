import tkinter as tk
from tkinter import ttk, messagebox
from environment.FarkelEnv import FarkleDQNEnv
from environment.tictactoe import TicTacToe
from environment.grid_word import GridWorld
from environment.line_word import LineWorld
from algos.DQN.ddqn import double_dqn_no_replay
from algos.DQN.ddqn_exp_replay import double_dqn_with_replay
from algos.DQN.deep_qlearning import deep_q_learning

class TrainingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("DRL Training Menu")
        self.master.geometry("800x600")
        self.master.configure(bg='#E0E0E0')

        # Variables
        self.env_var = tk.StringVar()
        self.algo_var = tk.StringVar()
        self.episodes_var = tk.IntVar(value=1000)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        self.batch_size_var = tk.IntVar(value=32)

        # Widgets
        self.create_widgets()

    def create_widgets(self):
        # Section Environnement
        tk.Label(self.master, text="Choisissez l'environnement :", font=('Helvetica', 16), bg='#E0E0E0').pack(pady=10)
        env_frame = tk.Frame(self.master, bg='#E0E0E0')
        env_frame.pack(pady=10)
        tk.Radiobutton(env_frame, text="Farkle", variable=self.env_var, value='farkle', bg='#E0E0E0').pack(side=tk.LEFT)
        tk.Radiobutton(env_frame, text="TicTacToe", variable=self.env_var, value='tictactoe', bg='#E0E0E0').pack(side=tk.LEFT)
        tk.Radiobutton(env_frame, text="GridWorld", variable=self.env_var, value='gridworld', bg='#E0E0E0').pack(side=tk.LEFT)
        tk.Radiobutton(env_frame, text="LineWorld", variable=self.env_var, value='lineworld', bg='#E0E0E0').pack(side=tk.LEFT)

        # Section Algorithmes
        tk.Label(self.master, text="Choisissez l'algorithme :", font=('Helvetica', 16), bg='#E0E0E0').pack(pady=10)
        algo_frame = tk.Frame(self.master, bg='#E0E0E0')
        algo_frame.pack(pady=10)
        tk.Radiobutton(algo_frame, text="DQN Classique", variable=self.algo_var, value='dqn', bg='#E0E0E0').pack(side=tk.LEFT)
        tk.Radiobutton(algo_frame, text="Double DQN (Sans Replay)", variable=self.algo_var, value='ddqn_no_replay', bg='#E0E0E0').pack(side=tk.LEFT)
        tk.Radiobutton(algo_frame, text="Double DQN (Avec Replay)", variable=self.algo_var, value='ddqn_with_replay', bg='#E0E0E0').pack(side=tk.LEFT)

        # Section Hyperparamètres
        tk.Label(self.master, text="Hyperparamètres :", font=('Helvetica', 16), bg='#E0E0E0').pack(pady=10)
        params_frame = tk.Frame(self.master, bg='#E0E0E0')
        params_frame.pack(pady=10)
        tk.Label(params_frame, text="Nombre d'épisodes :", bg='#E0E0E0').grid(row=0, column=0, padx=10)
        tk.Entry(params_frame, textvariable=self.episodes_var).grid(row=0, column=1, padx=10)
        tk.Label(params_frame, text="Taux d'apprentissage :", bg='#E0E0E0').grid(row=1, column=0, padx=10)
        tk.Entry(params_frame, textvariable=self.learning_rate_var).grid(row=1, column=1, padx=10)
        tk.Label(params_frame, text="Taille du batch :", bg='#E0E0E0').grid(row=2, column=0, padx=10)
        tk.Entry(params_frame, textvariable=self.batch_size_var).grid(row=2, column=1, padx=10)

        # Bouton de lancement
        tk.Button(self.master, text="Lancer l'entraînement", command=self.start_training, bg='#4CAF50', fg='white', font=('Helvetica', 14)).pack(pady=20)

        # Zone d'affichage des résultats
        self.result_text = tk.Text(self.master, height=10, bg='#F0F0F0')
        self.result_text.pack(pady=20)

    def start_training(self):
        """Lance l'entraînement en fonction des paramètres sélectionnés."""
        env_name = self.env_var.get()
        algo_name = self.algo_var.get()
        episodes = self.episodes_var.get()
        learning_rate = self.learning_rate_var.get()
        batch_size = self.batch_size_var.get()

        if not env_name or not algo_name:
            messagebox.showerror("Erreur", "Veuillez sélectionner un environnement et un algorithme.")
            return

        # Créer l'environnement
        env = self.create_environment(env_name)

        # Lancer l'algorithme choisi
        self.result_text.insert(tk.END, f"Lancement de l'entraînement : {algo_name} sur {env_name}\n")
        if algo_name == 'dqn':
            self.train_dqn(env, episodes, learning_rate, batch_size)
        elif algo_name == 'ddqn_no_replay':
            self.train_ddqn_no_replay(env, episodes, learning_rate)
        elif algo_name == 'ddqn_with_replay':
            self.train_ddqn_with_replay(env, episodes, learning_rate, batch_size)

    def create_environment(self, env_name):
        """Crée l'environnement en fonction du choix."""
        if env_name == 'farkle':
            return FarkleDQNEnv(target_score=2000)
        elif env_name == 'tictactoe':
            return TicTacToe()
        elif env_name == 'gridworld':
            return GridWorld(width=5, height=5)
        elif env_name == 'lineworld':
            return LineWorld(length=5)

    def train_dqn(self, env, episodes, learning_rate, batch_size):
        """Entraîne avec DQN classique."""
        self.result_text.insert(tk.END, "Démarrage de l'entraînement DQN...\n")
        # Implémentation DQN (simplifiée)
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=env.observation_space.shape[0]),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(env.action_space.n)
        ])
        deep_q_learning(model, env, episodes, learning_rate, batch_size)
        self.result_text.insert(tk.END, "Entraînement DQN terminé.\n")

    def train_ddqn_no_replay(self, env, episodes, learning_rate):
        """Entraîne avec Double DQN sans Replay."""
        self.result_text.insert(tk.END, "Démarrage de l'entraînement Double DQN (Sans Replay)...\n")
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=env.observation_space.shape[0]),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(env.action_space.n)
        ])
        double_dqn_no_replay(model, env, episodes, learning_rate)
        self.result_text.insert(tk.END, "Entraînement Double DQN (Sans Replay) terminé.\n")

    def train_ddqn_with_replay(self, env, episodes, learning_rate, batch_size):
        """Entraîne avec Double DQN avec Replay."""
        self.result_text.insert(tk.END, "Démarrage de l'entraînement Double DQN (Avec Replay)...\n")
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=env.observation_space.shape[0]),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(env.action_space.n)
        ])
        double_dqn_with_replay(model, env, episodes, learning_rate, batch_size)
        self.result_text.insert(tk.END, "Entraînement Double DQN (Avec Replay) terminé.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()
