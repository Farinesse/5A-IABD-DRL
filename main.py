import os
import sys
import time
import pyfiglet
from colorama import Fore, init
from typing import Optional, Dict

from tensorflow.python.keras.testing_utils import get_model_type

from algos.model_based.mtcs_utc import MCTS

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import environments
from environment.tictactoe import TicTacToe
from environment.FarkelEnv import FarkleDQNEnv
from environment.line_word import LineWorld
from environment.grid_word import GridWorld

from GUI.Farkel_GUI import main_gui
from GUI.test import load_model_pkl
from functions.outils import (
    human_move, play, play_agent_vs_random_tictactoe,
    play_grid_world, human_move_line_world,
    human_move_grid_world, play_line_grid_world, play_with_agent_gridworld, play_with_agent_lineworld, play_with_dqn,
    dqn_model_predict, play_with_reinforce, play_with_mcts
)
from functions.random import (
    random_agent_line_world, random_agent_grid_world
)


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
            "11": "1SAC",
            "12": "Random Rollout",
            "13": "MCTS (UCT)"
        }

        # Dictionnaire des chemins de modèles

        self.model_paths = {
            "Farkle": {
                "Deep Q-Learning": r"models/farkle/dqn_noreplay_farkle5000_e27c6866/dqn_noreplay_farkle5000_e27c6866.pkl",
                "Deep Q-Learning_with Experience Replay" : r"",
                "Double Deep Q-Learning": r"models/farkle/final_ddqn_noreplay_farkle5000ff092ddd_f355fe76/final_ddqn_noreplay_farkle5000ff092ddd_f355fe76.pkl",
                "Double Deep Q-Learning with Experience Replay": r"",
                "REINFORCE": r"models/farkle/reinforce_farkle_8417c0ab/reinforce_farkle_8417c0ab.pkl",
                "REINFORCE with Mean Baseline": r"models/farkle/reinforce_mb_farkle_918db84f/reinforce_mb_farkle_918db84f.pkl",
                "REINFORCE with Baseline Learned by a Critic": r"models/farkle/reinforce_mb_critic_farkle_9a85a598/reinforce_mb_critic_farkle_9a85a598.pkl",
                "PPO": r"models/farkle/ppo_farkle_1cdf12d9/ppo_farkle_1cdf12d9.pkl",
                "1SAC": r"models/farkle/1_step_actor_critic_farkle_22cc4053/1_step_actor_critic_farkle_22cc4053.pkl",
                "Random Rollout": r"",
                "MCTS (UCT)": r""
            },
            "TicTacToe": {
                "Deep Q-Learning": r"models/tictactoe/dqn_noreplay_tictactoe_13670294/dqn_noreplay_tictactoe_13670294.pkl",
                "Deep Q-Learning_with Experience Replay" : r"models/tictactoe/dqn_exp_replay_tictactoe_3add75b2/dqn_exp_replay_tictactoe_3add75b2.pkl",
                "Double Deep Q-Learning": r"models/tictactoe/ddqn_noreplay_tictactoe_2c91a69d/ddqn_noreplay_tictactoe_2c91a69d.pkl",
                "Double Deep Q-Learning with Experience Replay": r"models/tictactoe/ddqn_exp_replay_tictactoe_24375891/ddqn_exp_replay_tictactoe_24375891.pkl",
                "REINFORCE": r"models/tictactoe/reinforce_tictactoe_8b88b27b/reinforce_tictactoe_8b88b27b.pkl",
                "REINFORCE with Mean Baseline": r"models/tictactoe/reinforce_mb_tictactoe_ff9ef62c/reinforce_mb_tictactoe_ff9ef62c.pkl",
                "REINFORCE with Baseline Learned by a Critic": r"models/tictactoe/reinforce_mb_critic_tictactoe_98f48589/reinforce_mb_critic_tictactoe_98f48589.pkl",
                "PPO": r"models/tictactoe/ppo_tictactoe_8fbf74b6/ppo_tictactoe_8fbf74b6.pkl",
                "Random Rollout": r"",
                "MCTS (UCT)": r""
            },
            "LineWorld": {
                "Deep Q-Learning": r"models/line/dqn_noreplay_lineworld_0c06c177/dqn_noreplay_lineworld_0c06c177.pkl",
                "Deep Q-Learning_with Experience Replay": r"models/line/dqn_exp_replay_lineworld_e06bca3f/dqn_exp_replay_lineworld_e06bca3f.pkl",
                "Double Deep Q-Learning": r"models/line/ddqn_noreplay_lineworld_45554507/ddqn_noreplay_lineworld_45554507.pkl",
                "Double Deep Q-Learning with Experience Replay": r"models/line/ddqn_exp_replay_lineworld_9ee0e68b/ddqn_exp_replay_lineworld_9ee0e68b.pkl",
                "REINFORCE": r"models/line/reinforce_line_63d66199/reinforce_line_63d66199.pkl",
                "REINFORCE with Mean Baseline": r"models/line/reinforce_mb_line_3d2c5325/reinforce_mb_line_3d2c5325.pkl",
                "REINFORCE with Baseline Learned by a Critic": r"models/line/reinforce_mb_critic_line_0b1b03f9/reinforce_mb_critic_line_0b1b03f9.pkl",
                "PPO": r"models/line/ppo_line_fa5de30e/ppo_line_fa5de30e.pkl",
                "Random Rollout": r"",
                "MCTS (UCT)": r""
            },
            "GridWorld": {
                "Deep Q-Learning": r"environment/dqn_noreplay_gridworld_81bc8711/dqn_noreplay_gridworld_81bc8711.pkl",
                "Deep Q-Learning_with Experience Replay": r"models/grid/dqn_exp_replay_gridworld_2cfe1aa1/dqn_exp_replay_gridworld_2cfe1aa1.pkl",
                "Double Deep Q-Learning": r"models/grid/ddqn_noreplay_gridworld_33814660/ddqn_noreplay_gridworld_33814660.pkl",
                "Double Deep Q-Learning with Experience Replay": r"models/grid/ddqn_exp_replay_gridworld_24464756/ddqn_exp_replay_gridworld_24464756.pkl",
                "REINFORCE": r"models/grid/reinforce_grid_98956aa9/reinforce_grid_98956aa9.pkl",
                "REINFORCE with Mean Baseline": r"models/grid/reinforce_mb_critic_grid_4593302a/reinforce_mb_critic_grid_4593302a.pkl",
                "REINFORCE with Baseline Learned by a Critic": r"models/grid/reinforce_mb_grid_485edc35/reinforce_mb_grid_485edc35.pkl",
                "PPO": r"models/grid/ppo_grid_16274a2f/ppo_grid_16274a2f.pkl",
                "Random Rollout": r"",
                "MCTS (UCT)": r""
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
        line_world = LineWorld(length=10)
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
                                       "REINFORCE with Baseline Learned by a Critic", "PPO", "1SAC"]:
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
            if env == "Farkle":
                env = FarkleDQNEnv()
            elif env == "LineWorld":
                env = LineWorld(length=10)
            elif env == "GridWorld":
                env = GridWorld(width=5, height=5)
            elif env == "TicTacToe":
                env = TicTacToe()

            algo = self.choose_algorithm()
            if algo:
                print(Fore.CYAN + f"\nTest de {algo} dans l'environnement {env.__class__.__name__}")
                self.loading_animation("Préparation du test", 5)
                model_path = input("Chemain du model (format pkl): ")
                episodes = int(input("Nombre d'épisodes: "))
                if algo in [
                    "Tabular Q-Learning",
                    "Deep Q-Learning",
                    "Deep Q-Learning_with Experience Replay",
                    "Double Deep Q-Learning with Experience Replay",
                    "Double Deep Q-Learning"
                ]:
                    model = load_model_pkl(model_path)
                    mean_score, mean_time_per_episode, mean_steps_per_episode, mean_time_per_step, win_rate = play_with_dqn(env, model, dqn_model_predict, episodes=episodes)
                    print(
                        f"Score moyen: {mean_score}\n"
                        f"Temps moyen par épisode: {mean_time_per_episode}\n"
                        f"Nombre moyen de pas par épisode: {mean_steps_per_episode}\n"
                        f"Temps moyen par pas: {mean_time_per_step}\n"
                        f"Taux de victoire: {win_rate}"
                    )
                elif algo in [
                    "REINFORCE",
                    "REINFORCE with Mean Baseline",
                    "REINFORCE with Baseline Learned by a Critic",
                    "PPO",
                    "1SAC"
                ]:
                    model = load_model_pkl(model_path)
                    mean_score, mean_time_per_episode, mean_steps_per_episode, mean_time_per_step, win_rate = play_with_reinforce(env, model, None, episodes=episodes)
                    print(
                        f"Score moyen: {mean_score}\n"
                        f"Temps moyen par épisode: {mean_time_per_episode}\n"
                        f"Nombre moyen de pas par épisode: {mean_steps_per_episode}\n"
                        f"Temps moyen par pas: {mean_time_per_step}\n"
                        f"Taux de victoire: {win_rate}"
                    )
                elif algo == "MCTS (UCT)":
                    model = MCTS.load(model_path)
                    mean_score, mean_time_per_episode, mean_steps_per_episode, mean_time_per_step, win_rate = play_with_mcts(env, model, episodes=episodes)
                    print(
                        f"Score moyen: {mean_score}\n"
                        f"Temps moyen par épisode: {mean_time_per_episode}\n"
                        f"Nombre moyen de pas par épisode: {mean_steps_per_episode}\n"
                        f"Temps moyen par pas: {mean_time_per_step}\n"
                        f"Taux de victoire: {win_rate}"
                    )
                    pass

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