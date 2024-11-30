from play.play_farkle import play_farkle
#from play.play_tictactoe import play_tictactoe
#from utils.scores import load_statistics, display_statistics
from colorama import Fore, Style, init
import os

# Initialisation de Colorama
init(autoreset=True)

def clear_screen():
    """Nettoie l'écran pour une interface propre."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    """Affiche le menu principal."""
    clear_screen()
    print(Fore.GREEN + "=== MENU PRINCIPAL ===")
    print(Fore.CYAN + "1." + Fore.YELLOW + " Farkle")
    print(Fore.CYAN + "2." + Fore.YELLOW + " TicTacToe")
    print(Fore.CYAN + "3." + Fore.YELLOW + " GridWorld")
    print(Fore.CYAN + "4." + Fore.YELLOW + " LineWorld")
    print(Fore.CYAN + "5." + Fore.BLUE + " Afficher les statistiques")
    print(Fore.CYAN + "6." + Fore.RED + " Quitter")
    return input(Fore.BLUE + "Choisissez un environnement : " + Style.RESET_ALL)

def main():
    """Point d'entrée principal."""
    while True:
        choice = main_menu()

        if choice == "1":  # Farkle
            play_farkle()

        elif choice == "2":  # TicTacToe
            play_tictactoe()

        elif choice == "5":  # Statistiques
            scores = load_statistics()
            display_statistics(scores)

        elif choice == "6":  # Quitter
            print(Fore.GREEN + "Au revoir !")
            break

        else:
            print(Fore.RED + "Option invalide. Veuillez réessayer.")

if __name__ == "__main__":
    main()
