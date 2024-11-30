from Shell_Interface import main_menu
from play.play_farkle import farkle_menu
from utils.styles import print_colored



def main():
    """Point d'entrée principal du programme."""
    while True:
        choice = main_menu()
        if choice == "1":
            farkle_menu()  # Appelle le menu spécifique à Farkle
        elif choice == "2":
            print_colored("TicTacToe : Fonctionnalité à implémenter.", "red")
        elif choice == "3":
            print_colored("GridWorld : Fonctionnalité à implémenter.", "red")
        elif choice == "4":
            print_colored("LineWorld : Fonctionnalité à implémenter.", "red")
        elif choice == "5":
            print_colored("Au revoir !", "green")
            break
        else:
            print_colored("Option invalide. Réessayez.", "red")


if __name__ == "__main__":
    main()
