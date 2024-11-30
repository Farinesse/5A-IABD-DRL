import os
import time
import pyfiglet
from colorama import Fore, Style, init
from typing import Optional, Dict, Any

# Initialisation de colorama
init(autoreset=True)


class DRLInterface:
    def __init__(self):
        self.environments: Dict[str, str] = {
            "1": "TicTacToe",
            "2": "Farkle"
        }

        self.algorithms: Dict[str, str] = {
            "1": "Random",
            "2": "Tabular Q-Learning",
            "3": "Deep Q-Learning",
            "4": "Double Deep Q-Learning with Experience Replay",
            "5": "Double Deep Q-Learning",
            "6": "REINFORCE",
            "7": "REINFORCE with Mean Baseline",
            "8": "REINFORCE with Baseline Learned by a Critic",
            "9": "PPO",
            "10": "Random Rollout",
            "11": "MCTS (UCT)"
        }

    @staticmethod
    def clear_screen() -> None:
        """Nettoie l'écran du terminal."""
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def show_title(title: str) -> None:
        """Affiche un titre avec une police rétro."""
        ascii_art = pyfiglet.figlet_format(title)
        print(Fore.CYAN + ascii_art)

    @staticmethod
    def loading_animation(message: str, duration: int = 3) -> None:
        """Affiche une animation de chargement."""
        print(Fore.GREEN + message, end="", flush=True)
        for _ in range(duration):
            print(Fore.YELLOW + ".", end="", flush=True)
            time.sleep(0.5)
        print()

    def main_menu(self) -> str:
        """Affiche le menu principal et retourne le choix de l'utilisateur."""
        self.clear_screen()
        self.show_title("DRL Project")
        print(Fore.BLUE + "=== Menu Principal ===")
        print(Fore.MAGENTA + "1." + Fore.WHITE + " Jouer")
        print(Fore.MAGENTA + "2." + Fore.WHITE + " Entraîner un modèle")
        print(Fore.MAGENTA + "3." + Fore.WHITE + " Tester un modèle")
        print(Fore.MAGENTA + "4." + Fore.WHITE + " Quitter")
        return input(Fore.YELLOW + "\nChoisissez une option : ")

    def choose_environment(self) -> Optional[str]:
        """Interface de sélection de l'environnement."""
        print(Fore.BLUE + "\n=== Choisissez un Environnement ===")
        for key, env in self.environments.items():
            print(Fore.MAGENTA + f"{key}." + Fore.WHITE + f" {env}")
        print(Fore.MAGENTA + "3." + Fore.WHITE + " Retour")

        choice = input(Fore.YELLOW + "\nVotre choix : ")
        return self.environments.get(choice)

    def choose_algorithm(self) -> Optional[str]:
        """Interface de sélection de l'algorithme."""
        print(Fore.BLUE + "\n=== Choisissez un Algorithme ===")
        for key, algo in self.algorithms.items():
            print(Fore.MAGENTA + f"{key}." + Fore.WHITE + f" {algo}")
        print(Fore.MAGENTA + "12." + Fore.WHITE + " Retour")

        choice = input(Fore.YELLOW + "\nVotre choix : ")
        if choice == "12":
            return None
        return self.algorithms.get(choice)

    def run_play_mode(self) -> None:
        """Gestion du mode de jeu."""
        print(Fore.GREEN + "\n=== Mode Jouer ===")
        env = self.choose_environment()
        if env:
            print(Fore.CYAN + f"\nVous avez choisi de jouer dans : {env}")
            algo = self.choose_algorithm()
            if algo:
                print(Fore.CYAN + f"Algorithme sélectionné : {algo}")
                self.loading_animation("Préparation du jeu", 5)
                # TODO: Implémenter la logique de jeu ici

    def run_training_mode(self) -> None:
        """Gestion du mode d'entraînement."""
        print(Fore.GREEN + "\n=== Mode Entraînement ===")
        env = self.choose_environment()
        if env:
            algo = self.choose_algorithm()
            if algo:
                print(Fore.CYAN + f"\nEntraînement de {algo} dans l'environnement {env}")
                self.loading_animation("Démarrage de l'entraînement", 5)
                # TODO: Implémenter la logique d'entraînement ici

    def run_test_mode(self) -> None:
        """Gestion du mode de test."""
        print(Fore.GREEN + "\n=== Mode Test ===")
        env = self.choose_environment()
        if env:
            algo = self.choose_algorithm()
            if algo:
                print(Fore.CYAN + f"\nTest de {algo} dans l'environnement {env}")
                self.loading_animation("Préparation du test", 5)
                # TODO: Implémenter la logique de test ici

    def run(self) -> None:
        """Boucle principale de l'interface."""
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
    """Point d'entrée du programme."""
    interface = DRLInterface()
    interface.run()


if __name__ == "__main__":
    main()