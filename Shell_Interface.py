import os
import time
import pyfiglet
from colorama import Fore, Style, init

# Initialisation Colorama
init(autoreset=True)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_title(title):
    """Affiche un titre avec une police rétro"""
    ascii_art = pyfiglet.figlet_format(title)
    print(Fore.CYAN + ascii_art)

def loading_animation(message, duration=3):
    """Animation de chargement avec des points"""
    print(Fore.GREEN + message, end="", flush=True)
    for _ in range(duration):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)
    print()

def main_menu():
    """Affiche le menu principal"""
    clear_screen()
    show_title("DRL Project")
    print(Fore.BLUE + "=== Menu Principal ===")
    print(Fore.MAGENTA + "1." + Fore.WHITE + " Jouer")
    print(Fore.MAGENTA + "2." + Fore.WHITE + " Entraîner un modèle")
    print(Fore.MAGENTA + "3." + Fore.WHITE + " Tester un modèle")
    print(Fore.MAGENTA + "4." + Fore.WHITE + " Quitter")
    return input(Fore.YELLOW + "Choisissez une option : ")

def choose_environment():
    """Choix de l'environnement"""
    print(Fore.BLUE + "\n=== Choisissez un Environnement ===")
    print(Fore.MAGENTA + "1." + Fore.WHITE + " TicTacToe")
    print(Fore.MAGENTA + "2." + Fore.WHITE + " Farkle")
    print(Fore.MAGENTA + "3." + Fore.WHITE + " Retour")
    choice = input(Fore.YELLOW + "Votre choix : ")
    environments = { "1": "TicTacToe", "2": "Farkle" }
    return environments.get(choice, None)

def choose_algorithm():
    """Choix de l'algorithme"""
    print(Fore.BLUE + "\n=== Choisissez un Algorithme ===")
    print(Fore.MAGENTA + "1." + Fore.WHITE + " Random")
    print(Fore.MAGENTA + "2." + Fore.WHITE + " Deep Q-Learning")
    print(Fore.MAGENTA + "3." + Fore.WHITE + " Double Deep Q-Learning")
    print(Fore.MAGENTA + "4." + Fore.WHITE + " PPO")
    print(Fore.MAGENTA + "5." + Fore.WHITE + " Retour")
    choice = input(Fore.YELLOW + "Votre choix : ")
    algorithms = {
        "1": "Random",
        "2": "Deep Q-Learning",
        "3": "Double Deep Q-Learning",
        "4": "PPO"
    }
    return algorithms.get(choice, None)

def main():
    """Point d'entrée de l'interface"""
    while True:
        choice = main_menu()
        if choice == "1":
            print(Fore.GREEN + "=== Mode Jouer ===")
            env = choose_environment()
            if env:
                print(Fore.CYAN + f"Vous avez choisi de jouer dans : {env}")
                algo = choose_algorithm()
                print(Fore.CYAN + f"Algorithme sélectionné : {algo}")
                loading_animation("Préparation du jeu", 5)
        elif choice == "2":
            print(Fore.GREEN + "=== Mode Entraînement ===")
            env = choose_environment()
            algo = choose_algorithm()
            print(Fore.CYAN + f"Entraînement de {algo} dans l'environnement {env}")
            loading_animation("Démarrage de l'entraînement", 5)
        elif choice == "3":
            print(Fore.GREEN + "=== Mode Test ===")
            # Ajoute la logique de test ici
        elif choice == "4":
            print(Fore.RED + "Au revoir !")
            break
