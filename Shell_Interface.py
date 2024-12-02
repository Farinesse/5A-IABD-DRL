import os
import sys
import time
import pyfiglet
import tkinter as tk
import tensorflow as tf
import numpy as np
from colorama import Fore, Style, init
from typing import Optional, Dict, Any

from algos.model_based.mtcs_utc import MCTS

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import environments
from environment.tictactoe import TicTacToe
from environment.FarkelEnv import FarkleDQNEnv
from environment.line_word import LineWorld
from environment.grid_word import GridWorld

# Import GUI and utilities
from GUI.Farkel_GUI import main_gui
from GUI.test import predict_func, load_model_pkl
from functions.outils import (
    human_move, play, play_agent_vs_random_tictactoe,
    play_grid_world, human_move_line_world,
    human_move_grid_world, play_line_grid_world, play_with_agent_gridworld, play_with_agent_lineworld
)
from functions.random import (
    random_agent_line_world, random_agent_grid_world
)

# Initialisation de colorama
init(autoreset=True)


class DRLInterface:
    def __init__(self):
        self.environments: Dict[str, str] = {
            "1": "TicTacToe",
            "2": "Farkle",
            "3": "LineWorld",
            "4": "GridWorld"
        }

        self.algorithms: Dict[str, str] = {
            "1": "Random",
            "2": "Tabular Q-Learning",
            "3": "Deep Q-Learning",
            "4": "Deep Q-Learning_with Experience Replay",
            "5": "Double Deep Q-Learning with Experience Replay",
            "6": "Double Deep Q-Learning",
            "7": "REINFORCE",
            "8": "REINFORCE with Mean Baseline",
            "9": "REINFORCE with Baseline Learned by a Critic",
            "10": "PPO",
            "11": "Random Rollout",
            "12": "MCTS (UCT)"
        }

        # Dictionnaire des chemins de modèles

        self.model_paths = {
            "Farkle": {
                "Deep Q-Learning": r"models\farkle\dqn_noreplay_farkle5000_e27c6866.pkl",
                "Deep Q-Learning_with Experience Replay" : r"",
                "Double Deep Q-Learning": r"models\farkle\Double_Deep_Q_Learning.pkl",
                "Double Deep Q-Learning with Experience Replay": r"",
                "REINFORCE": r"",
                "REINFORCE with Mean Baseline": r"",
                "REINFORCE with Baseline Learned by a Critic": r"models/farkle/reinforce_mb_critic_farkle_9a85a598.pkl",
                "PPO": r"models/farkle/ppo_farkle_1cdf12d9.pkl",
                "Random Rollout": r"",
                "MCTS (UCT)": r""
            },
            "TicTacToe": {
                "Deep Q-Learning": r"models\TicTacToe\dqn_noreplay_tictactoe_13670294.pkl",
                "Deep Q-Learning_with Experience Replay" : r"models/TicTacToe/dqn_exp_replay_tictactoe_3add75b2.pkl",
                "Double Deep Q-Learning": r"models/TicTacToe/ddqn_noreplay_tictactoe_2c91a69d.pkl",
                "Double Deep Q-Learning with Experience Replay": r"models/TicTacToe/ddqn_exp_replay_tictactoe_f15bde40.pkl",
                "REINFORCE": r"algos/PolicyGradientMethods/tictactoe_reinforce_b2ebe91f/tictactoe_reinforce_b2ebe91f.pkl",
                "REINFORCE with Mean Baseline": r"algos/PolicyGradientMethods/tictactoe_reinforce_baseline_0d96bfaa/tictactoe_reinforce_baseline_0d96bfaa.pkl",
                "REINFORCE with Baseline Learned by a Critic": r"algos/PolicyGradientMethods/tictactoe_reinforce_b2ebe91f/tictactoe_reinforce_b2ebe91f.pkl",
                "PPO": r"",
                "Random Rollout": r"",
                "MCTS (UCT)": r"algos/model_based/models_/mcts_TicTacToe_100_sims_ep900.pkl"
            },
            "LineWorld": {
                "Deep Q-Learning": r"models/lineworld/dqn_noreplay_lineworld_0c06c177.pkl",
                "Deep Q-Learning_with Experience Replay": r"models/lineworld/dqn_exp_replay_lineworld_e06bca3f.pkl",
                "Double Deep Q-Learning": r"models/lineworld/ddqn_noreplay_lineworld_45554507.pkl",
                "Double Deep Q-Learning with Experience Replay": r"models/lineworld/ddqn_exp_replay_lineworld_9ee0e68b.pkl",
                "REINFORCE": r"",
                "REINFORCE with Mean Baseline": r"",
                "REINFORCE with Baseline Learned by a Critic": r"",
                "PPO": r"",
                "Random Rollout": r"",
                "MCTS (UCT)": r"algos/model_based/models/mcts_TicTacToe_100_sims_ep90.pkl"
            },
            "GridWorld": {
                "Deep Q-Learning": r"models/grid/dqn_noreplay_gridworld_1482d4bb.pkl",
                "Deep Q-Learning_with Experience Replay": r"models/grid/dqn_exp_replay_gridworld.h5_2cfe1aa1.pkl",
                "Double Deep Q-Learning": r"models/grid/ddqn_noreplay_gridworld_33814660.pkl",
                "Double Deep Q-Learning with Experience Replay": r"models/grid/ddqn_exp_replay_gridworld_24464756.pkl",
                "REINFORCE": r"",
                "REINFORCE with Mean Baseline": r"",
                "REINFORCE with Baseline Learned by a Critic": r"",
                "PPO": r"",
                "Random Rollout": r"",
                "MCTS (UCT)": r"algos/model_based/models_/mcts_GridWorld_100_sims_final.pkl"
            }
        }

    def get_model_path(self, algo: str, env_name: str) -> Optional[str]:
        """Retourne le chemin du modèle en fonction de l'algorithme et de l'environnement."""
        model_path = self.model_paths.get(env_name, {}).get(algo)
        if not model_path:
            print(Fore.RED + f"Erreur : Modèle pour {algo} dans {env_name} non défini.")
            return None
        if not os.path.exists(model_path):
            print(Fore.RED + f"Erreur : Modèle introuvable à l'emplacement {model_path}.")
            return None
        return model_path

    @staticmethod
    def clear_screen() -> None:
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def show_title(title: str) -> None:
        ascii_art = pyfiglet.figlet_format(title)
        print(Fore.CYAN + ascii_art)

    @staticmethod
    def loading_animation(message: str, duration: int = 3) -> None:
        print(Fore.GREEN + message, end="", flush=True)
        for _ in range(duration):
            print(Fore.YELLOW + ".", end="", flush=True)
            time.sleep(0.5)
        print()

    def main_menu(self) -> str:
        self.clear_screen()
        self.show_title("DRL Project")
        print(Fore.BLUE + "=== Menu Principal ===")
        print(Fore.MAGENTA + "1." + Fore.WHITE + " Jouer")
        print(Fore.MAGENTA + "2." + Fore.WHITE + " Entraîner un modèle")
        print(Fore.MAGENTA + "3." + Fore.WHITE + " Tester un modèle")
        print(Fore.MAGENTA + "4." + Fore.WHITE + " Quitter")
        return input(Fore.YELLOW + "\nChoisissez une option : ")

    def get_model_type(self, algo: str, model) -> str:
        """Détermine le type de modèle et affiche le résumé si nécessaire."""
        if algo == "MCTS (UCT)":
            return "mcts"
        elif algo in ["Deep Q-Learning", "Double Deep Q-Learning with Experience Replay",
                      "Deep Q-Learning_with Experience Replay", "Double Deep Q-Learning"]:
            print(model.summary())
            return "dqn"
        else:  # REINFORCE et variantes
            print(model.summary())
            return "reinforce"
    def choose_environment(self) -> Optional[str]:
        print(Fore.BLUE + "\n=== Choisissez un Environnement ===")
        for key, env in self.environments.items():
            print(Fore.MAGENTA + f"{key}." + Fore.WHITE + f" {env}")
        print(Fore.MAGENTA + "5." + Fore.WHITE + " Retour")
        choice = input(Fore.YELLOW + "\nVotre choix : ")
        return self.environments.get(choice)

    def play_farkle(self, player1_type: str, player2_type: str, path_model: Optional[str] = None, type_model = None) -> None:
        """Lance une partie de Farkle"""
        main_gui(
            player1_type=player1_type,
            player2_type=player2_type,
            path_model=path_model,
            type_model = type_model
        )
    def choose_algorithm(self) -> Optional[str]:
        print(Fore.BLUE + "\n=== Choisissez un Algorithme ===")
        for key, algo in self.algorithms.items():
            print(Fore.MAGENTA + f"{key}." + Fore.WHITE + f" {algo}")
        print(Fore.MAGENTA + "12." + Fore.WHITE + " Retour")
        choice = input(Fore.YELLOW + "\nVotre choix : ")
        if choice == "12":
            return None
        return self.algorithms.get(choice)

    def handle_lineworld(self) -> None:
        line_world = LineWorld(length=5)
        print(Fore.BLUE + "\n=== Choisissez le Type de Joueur ===")
        print(Fore.MAGENTA + "1." + Fore.WHITE + " Human")
        print(Fore.MAGENTA + "2." + Fore.WHITE + " Random")
        print(Fore.MAGENTA + "3." + Fore.WHITE + " Agent")
        print(Fore.MAGENTA + "4." + Fore.WHITE + " MCTS")

        mode = input(Fore.YELLOW + "\nVotre choix : ")
        if mode == "1":
            play_line_grid_world(line_world, player_human=human_move_line_world, print_game=True)
        elif mode == "2":
            play_line_grid_world(line_world, player_random=random_agent_line_world, print_game=True)
        elif mode in ["3", "4"]:
            if mode == "4":
                n_simulations = int(input(Fore.YELLOW + "Nombre de simulations MCTS (default: 100): ") or "100")
                model = MCTS.load(self.get_model_path("MCTS (UCT)", "LineWorld"))
                model.n_simulations = n_simulations
                type_model = "mcts"
            else:
                algo = self.choose_algorithm()
                if algo and algo != "Random":
                    model_path = self.get_model_path(algo, "LineWorld")
                    if os.path.exists(model_path):
                        model = load_model_pkl(model_path)
                        type_model = self.get_model_type(algo, model)

            play_with_agent_lineworld(line_world, model, num_games=100, type_model=type_model)

    def handle_gridworld(self) -> None:
        grid_world = GridWorld(width=5, height=5)
        print(Fore.BLUE + "\n=== Choisissez le Type de Joueur ===")
        print(Fore.MAGENTA + "1." + Fore.WHITE + " Human")
        print(Fore.MAGENTA + "2." + Fore.WHITE + " Random")
        print(Fore.MAGENTA + "3." + Fore.WHITE + " Agent")
        print(Fore.MAGENTA + "4." + Fore.WHITE + " MCTS")

        mode = input(Fore.YELLOW + "\nVotre choix : ")
        if mode == "1":
            play_line_grid_world(grid_world, player_human=human_move_grid_world, print_game=True)
        elif mode == "2":
            play_line_grid_world(grid_world, player_random=random_agent_grid_world, print_game=True)
        elif mode in ["3", "4"]:
            if mode == "4":
                n_simulations = int(input(Fore.YELLOW + "Nombre de simulations MCTS (default: 100): ") or "100")
                model = MCTS.load(self.get_model_path("MCTS (UCT)", "GridWorld"))
                model.n_simulations = n_simulations
                type_model = "mcts"
            else:
                algo = self.choose_algorithm()
                if algo and algo != "Random":
                    model_path = self.get_model_path(algo, "GridWorld")
                    if os.path.exists(model_path):
                        model = load_model_pkl(model_path)
                        type_model = self.get_model_type(algo, model)

            play_with_agent_gridworld(grid_world, model, num_games=100, type_model=type_model)

    def handle_farkle(self) -> None:
        print(Fore.BLUE + "\n=== Choisissez un Mode de Jeu ===")
        print(Fore.MAGENTA + "1." + Fore.WHITE + " Random vs Random")
        print(Fore.MAGENTA + "2." + Fore.WHITE + " Agent vs Random")
        print(Fore.MAGENTA + "3." + Fore.WHITE + " Human vs Agent")
        print(Fore.MAGENTA + "4." + Fore.WHITE + " Human vs Random")
        mode = input(Fore.YELLOW + "\nVotre choix : ")

        if mode == "1":
            self.play_farkle("random", "random")
        elif mode in ["2", "3"]:
            algo = self.choose_algorithm()
            if algo:
                model_path = self.get_model_path(algo, "Farkle")
                if model_path:  # Vérification du chemin valide
                    type_model = None
                    if algo in ["Deep Q-Learning", "Double Deep Q-Learning with Experience Replay",
                     "Deep Q-Learning_with Experience Replay", "Double Deep Q-Learning"] :
                        type_model = "dqn"
                    elif algo in ["REINFORCE", "REINFORCE with Mean Baseline",
                                       "REINFORCE with Baseline Learned by a Critic","PPO"]:
                        type_model = "reinforce"
                    else:
                        raise ValueError(f"Unknown algorithm: {algo}")
                    if mode == "2":
                        self.play_farkle("agent", "random", model_path,type_model)
                    elif mode == "3":
                        self.play_farkle("human", "agent", model_path,type_model)
                else:
                    print(Fore.YELLOW + "Utilisation du mode random à la place.")
                    self.play_farkle("random", "random")
        elif mode == "4":
            self.play_farkle("human", "random")

    def handle_tictactoe(self) -> None:
        tic_tac_toe = TicTacToe()
        print(Fore.BLUE + "\n=== Choisissez un Mode de Jeu ===")
        print(Fore.MAGENTA + "1." + Fore.WHITE + " Human vs Random")
        print(Fore.MAGENTA + "2." + Fore.WHITE + " Human vs Agent")
        print(Fore.MAGENTA + "3." + Fore.WHITE + " Human vs MCTS")

        mode = input(Fore.YELLOW + "\nVotre choix : ")
        if mode == "1":
            play(tic_tac_toe, human_move, lambda x: x.action_space.sample())
        elif mode in ["2", "3"]:
            if mode == "3":
                n_simulations = int(input(Fore.YELLOW + "Nombre de simulations MCTS (default: 100): ") or "100")
                model = MCTS.load(self.get_model_path("MCTS (UCT)", "TicTacToe"))
                model.n_simulations = n_simulations
                type_model = "mcts"
            else:
                algo = self.choose_algorithm()
                if algo:
                    model_path = self.get_model_path(algo, "TicTacToe")
                    if os.path.exists(model_path):
                        model = load_model_pkl(model_path)
                        type_model = self.get_model_type(algo, model)

            play_agent_vs_random_tictactoe(tic_tac_toe, model, num_games=100, type_model=type_model)

    def run_play_mode(self) -> None:
        print(Fore.GREEN + "\n=== Mode Jouer ===")
        env = self.choose_environment()

        if env == "LineWorld":
            self.handle_lineworld()
        elif env == "GridWorld":
            self.handle_gridworld()
        elif env == "Farkle":
            self.handle_farkle()
        elif env == "TicTacToe":
            self.handle_tictactoe()

    def run_training_mode(self) -> None:
        print(Fore.GREEN + "\n=== Mode Entraînement ===")
        env = self.choose_environment()
        if env:
            algo = self.choose_algorithm()
            if algo:
                print(Fore.CYAN + f"\nEntraînement de {algo} dans l'environnement {env}")
                self.loading_animation("Démarrage de l'entraînement", 5)
                # TODO: Implémenter la logique d'entraînement

    def run_test_mode(self) -> None:
        print(Fore.GREEN + "\n=== Mode Test ===")
        env = self.choose_environment()
        if env:
            algo = self.choose_algorithm()
            if algo:
                print(Fore.CYAN + f"\nTest de {algo} dans l'environnement {env}")
                self.loading_animation("Préparation du test", 5)
                # TODO: Implémenter la logique de test

    def run(self) -> None:
        while True:
            choice = self.main_menu()
            if choice == "1":
                self.run_play_mode()
            elif choice == "2":
                self.run_training_mode()
            elif choice == "3":
                self.run_test_mode()
            elif choice == "4":
                print(Fore.RED + "\nAu revoir !")
                break
            else:
                print(Fore.RED + "\nOption invalide. Veuillez réessayer.")
                time.sleep(1)


def main():
    interface = DRLInterface()
    interface.run()


if __name__ == "__main__":
    main()